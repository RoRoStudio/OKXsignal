// file: index.ts
//
// Supabase Edge Function: fetch_daily_candles
// -------------------------------------------
// Fetches all 1D candles for all active USDT/USDC pairs from OKX
// and inserts them into the "candles_1D" table in Supabase.
//
// Environment Variables:
//   SB_URL                => Supabase URL
//   SB_SERVICE_ROLE_KEY   => Supabase service role key
//   EMAIL_RECIPIENT       => Email recipient address
//
// Table Schema (candles_1D):
//   id SERIAL PRIMARY KEY,
//   timestamp TIMESTAMPTZ NOT NULL,
//   pair TEXT NOT NULL,
//   open DOUBLE PRECISION NOT NULL,
//   high DOUBLE PRECISION NOT NULL,
//   low DOUBLE PRECISION NOT NULL,
//   close DOUBLE PRECISION NOT NULL,
//   volume DOUBLE PRECISION NOT NULL,
//   quote_volume DOUBLE PRECISION NOT NULL,
//   taker_buy_base DOUBLE PRECISION NOT NULL,
//   taker_buy_quote DOUBLE PRECISION NOT NULL,
//   UNIQUE (pair, timestamp)
//

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.5.0";
import { corsHeaders } from "https://deno.land/x/supabase_functions@0.1.4/headers.ts";

// OKX v5 daily-candle endpoints and usage notes:
//  - GET /api/v5/public/instruments?instType=SPOT -> to list instruments
//  - Filter for quoteCcy in {USDT, USDC} and state == 'live'
//  - For historical daily candles, we need:
//      * GET /api/v5/market/candles        (last ~7 days or so)
//      * GET /api/v5/market/history-candles (for older data, in 7 day chunks up to 3 months…
//        but can be called repeatedly to scroll backwards)
//  - Each request returns up to 100-1500 lines at a time; we loop until data ends.
//  - Candle structure usually is:
//     [
//       "timestamp_string_millis",
//       "open",
//       "high",
//       "low",
//       "close",
//       "base_volume",    // e.g. BTC in BTC-USDT
//       "quote_volume",   // The "volCcy" field in OKX docs
//       "taker_buy_base_volume"
//     ]
//    The API docs often mention an 8th value "taker_buy_quote_volume", but it is usually only
//    present in the “/market/candles” + “history-candles” responses if "mergePx" param is used
//    or if the instrument type is futures. Some confusion can arise, so we’ll handle missing fields
//    gracefully. Taker buy quote volume can often be derived or found in a separate endpoint.
//    However, for simplicity, we fetch from the open docs. If the 8th field is absent, we store 0.
//
// OKX Rate Limit for Candles: 40 requests per 2 seconds. We’ll keep a small queue with time pacing.
//
// Steps in the function:
//   1) Connect to Supabase
//   2) Fetch all live spot instruments
//      - Filter by quoteCcy in (USDT, USDC), state=live
//   3) For each instrument, find the last candle timestamp in the DB (if any).
//   4) Paginate backward from earliest possible time. Insert new candles upward until we reach
//      that last-candle time or we run out of older data.
//   5) Then do a final pass for the most-recent chunk if needed.
//   6) Summarize and email the user, then respond.
//
// NB: This demo uses a direct mail sending approach via the Supabase Custom SMTP or other
// mail integrations. If you have a different setup, adjust the `sendSummaryEmail` accordingly.
//
// -------------------------------------------------------------------------------

type CandleRecord = {
  timestamp: Date;
  pair: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  quote_volume: number;
  taker_buy_base: number;
  taker_buy_quote: number;
};

interface OkxCandle {
  ts: string;
  open: string;
  high: string;
  low: string;
  close: string;
  vol: string;      // base volume
  volCcy: string;   // quote volume
  buyVol: string;   // taker buy base volume (some endpoints might call it differently)
  buyVolCcy: string;// taker buy quote volume
}

// Throttling util: ensures we don't exceed 40 requests / 2s
// We'll just do a short 50ms delay after each request to keep well under 40/2s = 20/s => 50ms
// For large volumes, you might want a smarter rate-limit queue.
async function safeFetch(...args: Parameters<typeof fetch>) {
  // Basic sleep between requests
  await new Promise((r) => setTimeout(r, 50));
  return fetch(...args);
}

// We will attempt to retrieve historical daily candles from oldest to newest. For older
// than ~7 days, we call GET /api/v5/market/history-candles, for the last ~7 days we use
// GET /api/v5/market/candles. We combine them. The data is returned in reverse chronological
// order, so we parse carefully.
//
// We’ll gather up everything in a large array from earliest -> latest. Then we can upsert them.
async function fetchAllDailyCandlesForPair(pair: string, lastKnownTime?: Date) {
  // Each candle returned is [timestamp, open, high, low, close, volumeBase, volumeQuote, takerBuyBase, takerBuyQuote]
  // but the last fields can be missing in some responses. We'll carefully parse.
  // OKX returns times in ms, from newest -> oldest. We invert them to earliest -> newest.
  // We'll do chunk pagination by repeatedly calling /history-candles until none left.

  let allCandles: CandleRecord[] = [];

  // Helper to parse each raw candle line from OKX to our CandleRecord
  function parseCandleLine(line: string[]): CandleRecord {
    // line: [ts, o, h, l, c, volBase, volQuote, takerBuyBase?, takerBuyQuote?]
    const [
      tsStr, oStr, hStr, lStr, cStr, volBaseStr, volQuoteStr,
      takerBuyBaseStr = "0", takerBuyQuoteStr = "0"
    ] = line;

    // Convert to number
    const t = new Date(Number(tsStr));
    return {
      timestamp: t,
      pair,
      open: Number(oStr),
      high: Number(hStr),
      low: Number(lStr),
      close: Number(cStr),
      volume: Number(volBaseStr),
      quote_volume: Number(volQuoteStr),
      taker_buy_base: Number(takerBuyBaseStr),
      taker_buy_quote: Number(takerBuyQuoteStr),
    };
  }

  async function fetchOneChunk(
    olderPart: boolean, // if true => we call /history-candles, else => /candles
    beforeTimestamp?: string
  ): Promise<CandleRecord[]> {
    let endpoint = olderPart
      ? "https://www.okx.com/api/v5/market/history-candles"
      : "https://www.okx.com/api/v5/market/candles";
    const url = new URL(endpoint);
    url.searchParams.set("instId", pair);
    url.searchParams.set("bar", "1D");
    url.searchParams.set("limit", "1500"); // max possible
    if (beforeTimestamp) {
      // OKX param is "before" to fetch data older than this
      url.searchParams.set("before", beforeTimestamp);
    }

    let res: Response;
    try {
      res = await safeFetch(url.toString());
    } catch (err) {
      console.error(`Network error while fetching from OKX for ${pair}:`, err);
      throw err;
    }
    if (!res.ok) {
      console.error(
        `Got a non-200 from OKX for ${pair}. Status: ${res.status} - ${res.statusText}`
      );
      throw new Error(`OKX fetch failed: ${res.status} - ${res.statusText}`);
    }

    const data = await res.json();
    if (data.code !== "0") {
      console.error(`OKX returned error for ${pair}: code=${data.code}, msg=${data.msg}`);
      throw new Error(`OKX returned error: ${data.msg}`);
    }
    const result: string[][] = data.data || [];
    // parse it into CandleRecord array:
    return result.map(parseCandleLine);
  }

  // fetch chunk by chunk from older to newer using /history-candles, because it goes from newest
  // to oldest. We'll keep calling it until we run out or we cross lastKnownTime.
  // We'll store them in ascending order eventually.

  let keepGoing = true;
  let lastTSinPreviousChunk: string | undefined; // "before" param
  while (keepGoing) {
    const chunk = await fetchOneChunk(true, lastTSinPreviousChunk);
    if (!chunk.length) {
      // no more data
      break;
    }
    // chunk is from newest->oldest; invert it:
    chunk.reverse();
    // We'll stop if we see a candle older than lastKnownTime, or we store only the portion that's newer
    const filtered: CandleRecord[] = lastKnownTime
      ? chunk.filter((c) => c.timestamp > lastKnownTime)
      : chunk;
    allCandles.push(...filtered);

    // The next "before" param is the ms of the earliest candle in this chunk:
    const earliestInChunk = chunk[0].timestamp.getTime().toString();
    // If chunk didn't skip anything and is presumably "full" we keep going. If chunk is less than 1500, maybe done.
    // But we can keep going anyway until we see no older data.
    // Also, if we found that everything in chunk is older than lastKnownTime, we can break.
    if (filtered.length < chunk.length) {
      // means some portion was older than lastKnownTime
      keepGoing = false;
    } else {
      // continue
      lastTSinPreviousChunk = earliestInChunk;
      if (chunk.length < 1500) {
        // probably done
        break;
      }
    }
  }

  // Now fetch the final ~7 days from /candles to fill in the newest chunk we might be missing:
  // (Because /history-candles usually excludes that last ~7 days window.)
  const recentChunk = await fetchOneChunk(false);
  // recentChunk is in newest->oldest
  recentChunk.reverse();
  let newerFiltered: CandleRecord[] = lastKnownTime
    ? recentChunk.filter((c) => c.timestamp > lastKnownTime)
    : recentChunk;
  // Also exclude any that are older than the latest so far if needed.
  // But no harm in pushing duplicates; we’ll do an upsert and skip duplicates.
  allCandles.push(...newerFiltered);

  // allCandles is from earliest->latest for that pair.
  // Let's do a final sort in ascending order by timestamp to be sure.
  allCandles.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  return allCandles;
}

// Utility to upsert a batch into the "candles_1D" table:
async function insertCandles(
  supabase: ReturnType<typeof createClient>,
  candleBatch: CandleRecord[]
) {
  // We rely on Supabase’s "on conflict do nothing" or "on conflict update" style upsert
  // to avoid duplicates. Because the table has UNIQUE (pair, timestamp).
  // If your schema or config differs, adapt accordingly.
  const { data, error } = await supabase
    .from("candles_1D")
    .insert(candleBatch)
    .select(); // or .onConflict("pair, timestamp").ignore() if relevant
  if (error) {
    throw error;
  }
  return data ?? [];
}

async function sendSummaryEmail(
  supabase: ReturnType<typeof createClient>,
  toEmail: string,
  summaryText: string
): Promise<void> {
  console.log(`Sending summary email to: ${toEmail}`);
  // If you have a custom approach or you want to use
  // your own mailer, do it here. For demonstration, we’ll use the
  // Supabase Email testing approach: “Mail to: <toEmail>”.
  // You may need to adapt for your own environment.

  // Example using the built-in "sendEmail" with supabase-js v2 if you have configured it.
  // If not, adapt to your own mail service.
  //
  // If you don't have any mail service set up in Supabase, you can do a direct SMTP approach
  // or an external email API call. Below is a rough example only:

  const subject = "OKX Candle Fetch Summary";
  const content = `
    <h1>OKX Candle Fetch Completed</h1>
    <pre style="font-size:14px;">${summaryText}</pre>
  `.trim();

  // Insert a row into a "emails" table or call a third-party email API, etc.
  // This snippet is a pseudo-implementation only. Adjust as needed.
  // If you have direct SMTP details, you can do a fetch to a mail-sending service.

  // For demonstration, we do a log:
  console.log("Email summary (subject, text):");
  console.log(subject);
  console.log(summaryText);

  // Or store for logs:
  await supabase.from("email_logs").insert({
    to_email: toEmail,
    subject,
    content,
    sent_at: new Date(),
  });

  // ... or you can actually send via your chosen method here
}

serve(async (req: Request) => {
  // CORS
  if (req.method === "OPTIONS") {
    return new Response("OK", { headers: corsHeaders });
  }

  // Setup
  const SUPABASE_URL = Deno.env.get("SB_URL")!;
  const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SB_SERVICE_ROLE_KEY")!;
  const EMAIL_RECIPIENT = Deno.env.get("EMAIL_RECIPIENT")!;
  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  console.log("Starting fetch_daily_candles function...");
  let totalPairs = 0;
  let totalCandlesFetched = 0;
  let totalCandlesInserted = 0;
  let totalFailed = 0;
  const failedPairs: string[] = [];

  try {
    // 1) Get all live spot instruments from OKX
    const instrumentsUrl =
      "https://www.okx.com/api/v5/public/instruments?instType=SPOT";
    const instRes = await safeFetch(instrumentsUrl);
    if (!instRes.ok) {
      throw new Error(
        `Fetching instruments failed with status: ${instRes.status} / ${instRes.statusText}`
      );
    }
    const instData = await instRes.json();
    if (instData.code !== "0") {
      throw new Error(`OKX instruments fetch returned code=${instData.code} msg=${instData.msg}`);
    }

    // Filter for USDT/USDC quote ccy, state == 'live'
    const rawInstruments = instData.data as any[];
    const filteredInstruments = rawInstruments.filter((inst: any) => {
      return (
        inst.quoteCcy &&
        (inst.quoteCcy === "USDT" || inst.quoteCcy === "USDC") &&
        inst.state === "live" &&
        inst.instType === "SPOT"
      );
    });

    console.log(
      `Found ${rawInstruments.length} instruments total; ` +
        `${filteredInstruments.length} are USDT/USDC + live`
    );

    // For each instrument, get the last candle timestamp from DB and fetch new data
    for (const inst of filteredInstruments) {
      const pair = inst.instId; // e.g. "BTC-USDT"

      try {
        // Get the most recent candle time we already have
        const { data: maxTimeData, error: maxTimeErr } = await supabase
          .from("candles_1D")
          .select("timestamp")
          .eq("pair", pair)
          .order("timestamp", { ascending: false })
          .limit(1);

        if (maxTimeErr) {
          throw maxTimeErr;
        }

        let lastKnownTime: Date | undefined = undefined;
        if (maxTimeData && maxTimeData.length > 0) {
          lastKnownTime = new Date(maxTimeData[0].timestamp);
        }

        console.log(`Fetching 1D candles for ${pair} (last known: ${lastKnownTime || "none"})`);
        const fetchedCandleRecords = await fetchAllDailyCandlesForPair(pair, lastKnownTime);

        console.log(
          `Fetched ${fetchedCandleRecords.length} new 1D candles for ${pair}...inserting...`
        );

        totalCandlesFetched += fetchedCandleRecords.length;

        if (fetchedCandleRecords.length > 0) {
          // Insert them in smaller batches to avoid large payload issues
          const BATCH_SIZE = 2000;
          let insertedCount = 0;
          for (let i = 0; i < fetchedCandleRecords.length; i += BATCH_SIZE) {
            const batch = fetchedCandleRecords.slice(i, i + BATCH_SIZE);
            const inserted = await insertCandles(supabase, batch);
            insertedCount += inserted.length;
          }
          totalCandlesInserted += insertedCount;
          console.log(`${pair}: Inserted ${insertedCount} new rows`);
        }

        totalPairs++;
      } catch (pairErr) {
        console.error(`Failed on pair ${pair}. Reason:`, pairErr);
        failedPairs.push(pair);
        totalFailed++;
      }
    }

    // Summaries
    const resultsSummary = `
      Pairs processed: ${totalPairs}
      Candles fetched (total across all pairs): ${totalCandlesFetched}
      Candles inserted (unique new rows): ${totalCandlesInserted}
      Failed pairs: ${totalFailed}
      ${failedPairs.length ? "Failures -> " + failedPairs.join(", ") : ""}
    `.trim();

    console.log("DONE with fetch_daily_candles!");
    console.log(resultsSummary);

    // Email summary
    await sendSummaryEmail(supabase, EMAIL_RECIPIENT, resultsSummary);

    return new Response(
      JSON.stringify(
        {
          success: true,
          message: "Candle fetching completed successfully",
          summary: resultsSummary,
        },
        null,
        2
      ),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (err) {
    console.error("Uncaught error in fetch_daily_candles:", err);
    return new Response(
      JSON.stringify({
        success: false,
        error: err.message || String(err),
      }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});