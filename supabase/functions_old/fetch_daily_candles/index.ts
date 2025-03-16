/*********************************************************************************************
  A MINIMAL SUPABASE EDGE FUNCTION THAT JUST WORKS
  Deno environment, no Node imports.

  Flow:
    1) fetchInstruments() -> all USDT/USDC "live" spot pairs
    2) For each pair:
       - getLastStoredTimestamp() => lastTS
       - fetchAllCandlesDescending(pair, lastTS)
         => repeated calls with "before=someTs"
         => gather all daily candles older than that 'before'
         => break if we get 0 or fewer than 100
       - skip candles whose ts <= lastTS (duplicates)
       - sort ascending
       - insert

  TABLE candles_1D (unique on pair,timestamp):
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    pair TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    quote_volume DOUBLE PRECISION NOT NULL,
    taker_buy_base DOUBLE PRECISION NOT NULL,
    taker_buy_quote DOUBLE PRECISION NOT NULL,
    UNIQUE (pair, timestamp)

  WARNINGS:
    • This code fetches *all* historical daily candles for each pair. 
      For a big number of pairs and many years, that's a lot of requests.
    • If you see "0 data" from OKX, we log the raw JSON so you can confirm the reason.

*********************************************************************************************/

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// ENV
const SUPABASE_URL              = Deno.env.get("SB_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SB_SERVICE_ROLE_KEY")!;
const EMAIL_RECIPIENT           = Deno.env.get("EMAIL_RECIPIENT")!;

// OKX constants
const OKX_API_URL           = "https://www.okx.com/api/v5/market/candles";
const OKX_INSTRUMENTS_URL   = "https://www.okx.com/api/v5/public/instruments?instType=SPOT";
const TIMEFRAME             = "1D";
const MAX_CANDLES_PER_CHUNK = 100;

// We'll add a small 200ms sleep after each request => ~5 requests/sec
async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// MAIN ENTRY
Deno.serve(async (req) => {
  console.log("▶ Starting minimal daily candle fetch...");

  try {
    // 1) Get instruments (USDT/USDC)
    const instruments = await fetchInstruments();
    if (!instruments.length) {
      throw new Error("No USDT/USDC instruments found");
    }
    console.log(`✅ Found ${instruments.length} instruments`);

    // 2) For each pair => fetch & store
    let totalInserted = 0;
    const failedPairs: string[] = [];
    for (const pair of instruments) {
      try {
        const inserted = await syncPair(pair);
        totalInserted += inserted;
        console.log(`✅ [${pair}] => Inserted ${inserted} new candles`);
      } catch (err) {
        console.error(`❌ [${pair}] => FAILED =>`, err);
        failedPairs.push(pair);
      }
    }

    // 3) Email
    await sendEmailReport(totalInserted, failedPairs);
    console.log("✅ Completed daily candle fetch. Inserted total:", totalInserted);

    return new Response(
      JSON.stringify({ success: true, inserted: totalInserted, failed: failedPairs }), 
      { status: 200, headers: { "Content-Type": "application/json" } }
    );

  } catch (error) {
    console.error("🚨 Fatal error =>", error);
    await sendEmailReport(0, [], error.message);
    return new Response(
      JSON.stringify({ success: false, error: error.message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
});

// fetchInstruments => returns all live spot pairs that end in -USDT or -USDC
async function fetchInstruments(): Promise<string[]> {
  console.log("▶ fetchInstruments => checking OKX /public/instruments for USDT/USDC...");
  try {
    const resp = await fetch(OKX_INSTRUMENTS_URL);
    await sleep(200); // rate-limit
    if (!resp.ok) {
      console.warn("⚠️ fetchInstruments => non-200 =>", resp.status);
      return [];
    }
    const json = await resp.json() as { data?: any[] };
    if (!json?.data) {
      console.warn("⚠️ fetchInstruments => missing data =>", json);
      return [];
    }
    const arr = json.data;
    // filter "instId" ends in -USDT or -USDC, and state=live
    const pairs = arr
      .filter(d => 
        (d.instId.endsWith("-USDT") || d.instId.endsWith("-USDC")) 
        && d.state === "live"
      )
      .map(d => d.instId);
    return pairs;
  } catch (err) {
    console.error("❌ fetchInstruments => error =>", err);
    return [];
  }
}

// syncPair => high-level
async function syncPair(pair: string): Promise<number> {
  const lastStored = await getLastStoredTimestamp(pair);
  console.log(`▶ [${pair}] lastStored => ${lastStored} => ${new Date(lastStored).toISOString()}`);

  // if lastStored=0 => that means we never stored a candle for this pair
  // so we'll do "before = Date.now()" for first chunk
  const newCandles = await fetchAllCandlesDescending(pair, lastStored);
  if (!newCandles.length) {
    console.log(`ℹ️ [${pair}] => no new candles => skipping insert`);
    return 0;
  }
  // Insert
  await storeCandles(pair, newCandles);
  return newCandles.length;
}

// getLastStoredTimestamp => newest candle in DB
async function getLastStoredTimestamp(pair: string): Promise<number> {
  try {
    const { data, error } = await supabase
      .from("candles_1D")
      .select("timestamp")
      .eq("pair", pair)
      .order("timestamp", { ascending: false })
      .limit(1);

    if (error) {
      console.warn(`⚠️ [${pair}] getLastStoredTimestamp =>`, error);
      return 0;
    }
    if (!data || !data.length) return 0;
    return new Date(data[0].timestamp).getTime();
  } catch (err) {
    console.warn(`⚠️ [${pair}] getLastStoredTimestamp => catch =>`, err);
    return 0;
  }
}

// fetchAllCandlesDescending => repeated calls in descending order
// start: if lastStored=0 => before= Date.now()
// else => before= lastStored
// we keep chunking until we get < 100 or 0 data
async function fetchAllCandlesDescending(pair: string, lastStored: number): Promise<any[]> {
  console.log(`▶ [${pair}] fetchAllCandlesDescending => lastStored=${lastStored}`);
  const results: any[] = [];

  // If we never stored anything => we'll start from "now"
  let currentBefore = (lastStored === 0) ? Date.now() : lastStored;
  let chunkCount = 0;

  while (true) {
    const beforeParam = Math.floor(currentBefore / 1000) * 1000;
    const url = `${OKX_API_URL}?instId=${pair}&bar=${TIMEFRAME}&limit=${MAX_CANDLES_PER_CHUNK}&before=${beforeParam}`;
    console.log(`🔄 [${pair}] chunk #${chunkCount+1} => before=${beforeParam} => ${new Date(beforeParam).toISOString()}`);

    let raw;
    try {
      const resp = await fetch(url);
      await sleep(200); // rate-limit
      if (!resp.ok) {
        console.warn(`⚠️ [${pair}] chunk #${chunkCount+1} => non-200 =>`, resp.status);
        break;
      }
      raw = await resp.json();
    } catch (err) {
      console.warn(`⚠️ [${pair}] chunk #${chunkCount+1} => fetch error =>`, err);
      break;
    }

    if (!raw?.data) {
      console.warn(`⚠️ [${pair}] chunk #${chunkCount+1} => no 'data' =>`, JSON.stringify(raw).slice(0,400));
      break;
    }

    const arr = raw.data;
    console.log(`ℹ️ [${pair}] chunk #${chunkCount+1} => got ${arr.length} candles total. Example:`, arr[0] || "none");

    if (!arr.length) {
      // no data => done
      console.warn(`⚠️ [${pair}] chunk #${chunkCount+1} => 0 data => break`);
      break;
    }

    let oldestTsInChunk = currentBefore;
    let accepted = 0;

    // Each candle: [ts, open, high, low, close, volume, quoteVol, takerBuyBase, takerBuyQuote]
    // They come newest -> oldest
    for (const c of arr) {
      if (!Array.isArray(c) || c.length<9) {
        continue; // skip malformed
      }
      const [tsStr, open, high, low, close, vol, qVol, takerBase, takerQuote] = c;
      const tsNum = parseInt(tsStr, 10);

      if (tsNum <= lastStored) {
        // means we already have it => skip
        continue;
      }
      // Accept
      results.push({
        timestamp      : new Date(tsNum).toISOString(),
        pair,
        open           : +open,
        high           : +high,
        low            : +low,
        close          : +close,
        volume         : +vol,
        quote_volume   : +qVol,
        taker_buy_base : +takerBase,
        taker_buy_quote: +takerQuote,
      });
      oldestTsInChunk = Math.min(oldestTsInChunk, tsNum);
      accepted++;
    }

    console.log(`✅ [${pair}] chunk #${chunkCount+1} => accepted ${accepted} / ${arr.length}`);

    // next chunk => before = oldestTsInChunk - 1
    currentBefore = oldestTsInChunk - 1;
    chunkCount++;

    if (arr.length < MAX_CANDLES_PER_CHUNK) {
      // probably no more older data
      console.log(`ℹ️ [${pair}] chunk #${chunkCount} => got <100 => done`);
      break;
    }
  }

  // final sort ascending
  results.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  console.log(`➡️ [${pair}] total final new candles => ${results.length}`);
  return results;
}

// storeCandles => just do an insert
async function storeCandles(pair: string, candles: any[]) {
  if (!candles.length) {
    console.log(`ℹ️ [${pair}] => storeCandles => 0 => skipping`);
    return;
  }
  try {
    const { error } = await supabase
      .from("candles_1D")
      .insert(candles);
    if (error) {
      console.error(`❌ [${pair}] insert error =>`, error);
    } else {
      console.log(`✅ [${pair}] Inserted ${candles.length} candles into DB`);
    }
  } catch (err) {
    console.error(`❌ [${pair}] storeCandles =>`, err);
  }
}

// sendEmailReport => minimal
async function sendEmailReport(inserted: number, failed: string[], errorMsg = "") {
  const now = new Date().toISOString();
  const text = `
Daily Candle Sync
Time: ${now}
Inserted: ${inserted}
Failed pairs: ${failed.length ? failed.join(", ") : "none"}
${errorMsg ? "ERROR: " + errorMsg : ""}
`;
  try {
    const resp = await fetch(`${SUPABASE_URL}/functions/v1/send-email`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`
      },
      body: JSON.stringify({
        to: EMAIL_RECIPIENT,
        subject: "OKX Candle Sync Results",
        text
      })
    });
    if (!resp.ok) {
      console.warn(`⚠️ sendEmailReport => non-200 => ${resp.status}`);
    }
  } catch (err) {
    console.error("❌ sendEmailReport =>", err);
  }
}
