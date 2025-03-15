/**********************************************************************
    SUPABASE EDGE FUNCTION WITH OKX RATE LIMIT HANDLING
    DENO environment, no Node imports.

    Flow:
      1)  fetchInstruments() -> all USDT/USDC spot pairs from OKX
      2)  For each pair -> getLastStoredTimestamp() -> fetchAllCandles() -> storeCandles()
      3)  Summaries + error handling -> sendEmailReport

    Rate-limit logic:
      - OKX says 40 requests per 2 seconds for the relevant endpoint.
      - We'll do a small pause (e.g. 200ms) after every single request to keep it ~5 requests/sec,
        which is definitely under 40/2sec.

**********************************************************************/

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// ─────────────────────────────────────────────────────────────
//  ENV Vars
// ─────────────────────────────────────────────────────────────
const SUPABASE_URL               = Deno.env.get("SB_URL")!;
const SUPABASE_SERVICE_ROLE_KEY  = Deno.env.get("SB_SERVICE_ROLE_KEY")!;
const EMAIL_RECIPIENT            = Deno.env.get("EMAIL_RECIPIENT")!;

// OKX constants
const OKX_API_URL          = "https://www.okx.com/api/v5/market/candles";
const OKX_INSTRUMENTS_URL  = "https://www.okx.com/api/v5/public/instruments?instType=SPOT";
const TIMEFRAME            = "1D";
const MAX_CANDLES_PER_CALL = 100;

// We'll define a "hard" cutoff date in ms, so we don't go back infinitely.
const HARDCUTOFF_MS = new Date("2017-01-01T00:00:00Z").getTime();

// Insert a short sleep after each request to respect rate limit
// 200ms => 5 requests/second => 10 requests in 2s => well below 40 in 2s
async function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─────────────────────────────────────────────────────────────
//  Supabase client
// ─────────────────────────────────────────────────────────────
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// ─────────────────────────────────────────────────────────────
//  MAIN ENTRY POINT
// ─────────────────────────────────────────────────────────────
Deno.serve(async (req) => {
  console.log("▶ Starting daily candle sync (with OKX rate limit handling) ...");

  try {
    const instruments = await fetchInstruments(); 
    if (!instruments.length) {
      throw new Error("No USDT/USDC spot pairs found on OKX");
    }
    console.log(`✅ Found ${instruments.length} relevant pairs.`);

    let totalInserted = 0;
    let failedPairs: string[] = [];

    for (const pair of instruments) {
      try {
        const insertedCount = await syncPair(pair);
        totalInserted += insertedCount;
        console.log(`✅ [${pair}] Inserted ${insertedCount} new candles`);
      } catch (err) {
        console.error(`❌ [${pair}] Failure:`, err);
        failedPairs.push(pair);
      }
    }

    // Send final email
    await sendEmailReport(totalInserted, failedPairs);
    console.log(`✅ Sync complete. Inserted total of ${totalInserted} candles.`);

    return new Response(JSON.stringify({ 
      success: true, 
      inserted: totalInserted, 
      failed: failedPairs 
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });

  } catch (error) {
    console.error("🚨 Fatal error in daily candle sync:", error);
    await sendEmailReport(0, [], error.message);
    return new Response(JSON.stringify({ 
      success: false, 
      error: error.message 
    }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
});

// ─────────────────────────────────────────────────────────────
//  fetchInstruments()
//    - Get all USDT/USDC "live" spot pairs from OKX
//    - Rate-limit: Just one call, then we do a 200ms sleep
// ─────────────────────────────────────────────────────────────
async function fetchInstruments(): Promise<string[]> {
  console.log("▶ Fetching USDT & USDC spot instruments from OKX...");
  try {
    const resp = await Promise.race([
      fetch(OKX_INSTRUMENTS_URL),
      new Promise((_, reject) => setTimeout(() => reject(new Error("⏳ OKX Timeout!")), 5000))
    ]) as Response;
    // Sleep after the request to avoid spamming (one request here).
    await sleep(200);

    if (!resp.ok) {
      console.warn("⚠️ Non-200 fetching instruments. Status:", resp.status);
      return [];
    }
    const data = await resp.json() as { data?: any[]; code?: string; msg?: string; };
    if (!data.data) {
      console.warn("⚠️ instruments API returned empty data");
      return [];
    }
    const instruments = data.data
      .filter(d => (d.instId.endsWith("-USDT") || d.instId.endsWith("-USDC")) && d.state === "live")
      .map(d => d.instId);
    console.log(`✅ fetchInstruments() => ${instruments.length} instruments`);
    return instruments;
  } catch (err) {
    console.error("❌ Error fetching instruments:", err);
    return [];
  }
}

// ─────────────────────────────────────────────────────────────
//  getLastStoredTimestamp(pair)
//    - newest candle in "candles_1D", or 0 if none
// ─────────────────────────────────────────────────────────────
async function getLastStoredTimestamp(pair: string): Promise<number> {
  try {
    const { data, error } = await supabase
      .from("candles_1D")
      .select("timestamp")
      .eq("pair", pair)
      .order("timestamp", { ascending: false })
      .limit(1);

    if (error) {
      console.warn(`⚠️ [${pair}] getLastStoredTimestamp error:`, error);
      return 0;
    }
    if (!data || !data.length) {
      return 0;
    }
    return new Date(data[0].timestamp).getTime();
  } catch (err) {
    console.warn(`⚠️ Unexpected error in getLastStoredTimestamp for [${pair}]:`, err);
    return 0;
  }
}

// ─────────────────────────────────────────────────────────────
//  fetchAllCandles(pair, lastTimestamp)
//    - fetch from OKX in descending order
//    - keep stepping older until no data or cutoff date
//    - collect everything strictly newer than lastTimestamp
//    - return ascending
// ─────────────────────────────────────────────────────────────
async function fetchAllCandles(pair: string, lastTimestamp: number): Promise<any[]> {
  const allCandles: any[] = [];
  // Start from now if lastTS=0, or from lastTS if we do have it:
  let currentBefore = lastTimestamp > 0 ? lastTimestamp : Date.now();

  while (true) {
    if (currentBefore < HARDCUTOFF_MS) {
      console.log(`⚠️ [${pair}] Reached cutoff date. Stopping.`);
      break;
    }

    const useBefore = Math.floor(currentBefore / 1000) * 1000;
    const url = `${OKX_API_URL}?instId=${pair}&bar=${TIMEFRAME}&limit=${MAX_CANDLES_PER_CALL}&before=${useBefore}`;
    console.log(`🔄 [${pair}] Fetch chunk => ${url}`);

    let jsonData: any;
    try {
      const resp = await Promise.race([
        fetch(url),
        new Promise((_, reject) => setTimeout(() => reject(new Error("⏳ OKX Timeout!")), 5000))
      ]) as Response;

      // Sleep after the request to keep under rate limit
      await sleep(200);

      if (!resp.ok) {
        console.warn(`⚠️ [${pair}] Non-OK response: ${resp.status}`);
        break;
      }
      jsonData = await resp.json();
    } catch (err) {
      console.warn(`⚠️ [${pair}] Error fetching chunk:`, err);
      break;
    }

    if (!jsonData?.data || !jsonData.data.length) {
      console.warn(`⚠️ [${pair}] No more data from OKX.`);
      break;
    }

    let oldestInThisBatch = currentBefore;
    let newCandlesCount   = 0;

    for (const c of jsonData.data) {
      if (c.length < 9) {
        continue; // malformed
      }
      const ts = parseInt(c[0], 10);
      if (ts <= lastTimestamp) {
        // already have it
        continue;
      }
      if (ts < HARDCUTOFF_MS) {
        // older than our global cutoff => skip
        continue;
      }
      allCandles.push({
        timestamp      : new Date(ts).toISOString(),
        pair           : pair,
        open           : parseFloat(c[1]),
        high           : parseFloat(c[2]),
        low            : parseFloat(c[3]),
        close          : parseFloat(c[4]),
        volume         : parseFloat(c[5]),
        quote_volume   : parseFloat(c[6]),
        taker_buy_base : parseFloat(c[7]),
        taker_buy_quote: parseFloat(c[8]),
      });
      oldestInThisBatch = Math.min(oldestInThisBatch, ts);
      newCandlesCount++;
    }

    console.log(`✅ [${pair}] chunk had ${jsonData.data.length} raw, accepted ${newCandlesCount}`);

    currentBefore = oldestInThisBatch - 1;

    if (jsonData.data.length < MAX_CANDLES_PER_CALL) {
      console.log(`ℹ️ [${pair}] Fewer than ${MAX_CANDLES_PER_CALL} => done.`);
      break;
    }
  }

  // Sort ascending 
  allCandles.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  return allCandles;
}

// ─────────────────────────────────────────────────────────────
//  storeCandles(pair, candles)
//    - Insert new rows into the DB
// ─────────────────────────────────────────────────────────────
async function storeCandles(pair: string, candles: any[]) {
  if (!candles.length) {
    console.log(`ℹ️ [${pair}] No new candles to insert.`);
    return;
  }
  try {
    const { error } = await supabase
      .from("candles_1D")
      .insert(candles);

    if (error) {
      console.error(`❌ [${pair}] Insert error:`, error);
    } else {
      console.log(`✅ [${pair}] Inserted ${candles.length} new candles.`);
    }
  } catch (err) {
    console.error(`❌ [${pair}] Unexpected insert error:`, err);
  }
}

// ─────────────────────────────────────────────────────────────
//  syncPair(pair)
//    - orchestrates the process for one pair
// ─────────────────────────────────────────────────────────────
async function syncPair(pair: string): Promise<number> {
  const lastTS = await getLastStoredTimestamp(pair);
  console.log(`▶ [${pair}] last stored => ${new Date(lastTS).toISOString()}`);

  const fetched = await fetchAllCandles(pair, lastTS);
  if (!fetched.length) {
    await storeCandles(pair, []); // will log "No new candles"
    return 0;
  }
  await storeCandles(pair, fetched);
  return fetched.length;
}

// ─────────────────────────────────────────────────────────────
//  sendEmailReport(inserted, failedPairs, errorMsg?)
// ─────────────────────────────────────────────────────────────
async function sendEmailReport(
  inserted: number, 
  failedPairs: string[],
  errorMsg = ""
) {
  const nowIso = new Date().toISOString();
  const textBody = `
    ✅ Candle Sync @ ${nowIso}
    => Inserted: ${inserted}
    => Failed: ${failedPairs.length}
    ${failedPairs.length ? "🔴 " + failedPairs.join(", ") : ""}
    ${errorMsg ? `\n⚠️  Errors: ${errorMsg}` : ""}
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
        subject: "📊 Daily Candle Sync Results",
        text: textBody
      })
    });
    if (!resp.ok) {
      console.warn("⚠️ Email function responded non-200:", resp.status);
    }
  } catch (err) {
    console.error("❌ Error sending email:", err);
  }
}
