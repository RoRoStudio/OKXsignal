// ✅ Load Supabase Edge Runtime Definitions
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// ✅ Load environment variables from Deno
const SUPABASE_URL = Deno.env.get("SB_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SB_SERVICE_ROLE_KEY")!;
const EMAIL_RECIPIENT = Deno.env.get("EMAIL_RECIPIENT")!;
const OKX_API_URL = "https://www.okx.com/api/v5/market/candles";
const OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT";
const MAX_CANDLES = 100;
const TIMEFRAME = "1D";
const RETRY_LIMIT = 3;

// ✅ Initialize Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// ✅ Supabase Edge Function Entry Point
Deno.serve(async (req) => {
    console.log("📡 Background Task: Fetching daily market data...");

    try {
        const instruments = await fetchInstruments();
        if (!instruments.length) throw new Error("No active USDT/USDC pairs found!");

        console.log(`✅ ${instruments.length} trading pairs fetched.`);
        
        let totalInserted = 0;
        let failedPairs: string[] = [];

        for (const pair of instruments) {
            try {
                const insertedCount = await syncPairData(pair);
                totalInserted += insertedCount;
                console.log(`✅ Inserted ${insertedCount} candles for ${pair}`);
            } catch (error) {
                console.error(`❌ Failed for ${pair}:`, error);
                failedPairs.push(pair);
            }
        }

        await sendEmailReport(totalInserted, failedPairs);
        console.log(`✅ Sync complete: ${totalInserted} candles inserted.`);

        return new Response(JSON.stringify({ success: true, inserted: totalInserted, failed: failedPairs.length }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
        });
    } catch (error) {
        console.error("🚨 Fatal Error:", error);
        await sendEmailReport(0, [], error.message);
        return new Response(JSON.stringify({ success: false, error: error.message }), {
            status: 500,
            headers: { "Content-Type": "application/json" },
        });
    }
});

// ✅ Fetch all active USDT & USDC trading pairs
async function fetchInstruments(): Promise<string[]> {
    console.log("📊 Fetching available USDT & USDC spot trading pairs...");
    
    try {
        const response = await fetch(OKX_INSTRUMENTS_URL);
        const data = await response.json() as { data?: any[] };

        if (!data.data) {
            console.warn("⚠️ No trading pairs found in API response.");
            return [];
        }

        const pairs = data.data
            .filter(x => (x.instId.endsWith("-USDT") || x.instId.endsWith("-USDC")) && x.state === "live")
            .map(x => x.instId);

        console.log(`✅ Found ${pairs.length} trading pairs.`);
        return pairs;
    } catch (error) {
        console.error("❌ Error fetching instruments:", error);
        return [];
    }
}

// ✅ Get last stored timestamp to fetch only missing candles
async function getLastStoredTimestamp(pair: string): Promise<number> {
    try {
        const { data, error } = await supabase
            .from("candles_1D")
            .select("timestamp")
            .eq("pair", pair)
            .order("timestamp", { ascending: false })
            .limit(1);

        if (error) {
            console.warn(`⚠️ Error fetching last timestamp for ${pair}:`, error);
            return 0;
        }

        return data.length ? new Date(data[0].timestamp).getTime() : 0;
    } catch (error) {
        console.warn(`⚠️ Unexpected error fetching timestamp for ${pair}:`, error);
        return 0;
    }
}

// ✅ Fetch missing candles from OKX with retries + logging + timestamp correction
async function fetchCandles(pair: string, lastTimestamp: number): Promise<any[]> {
  console.log(`📡 Fetching candles for ${pair} (since ${new Date(lastTimestamp).toISOString()})`);

  let candles: any[] = [];
  let startTime = lastTimestamp || Date.now(); // Start from now if no data
  let attempts = 0;

  while (startTime > 1483228800000 && attempts < RETRY_LIMIT) { // Stop if older than 2017
      try {
          // ✅ Convert `before` timestamp to correct format
          const beforeTimestamp = Math.floor(startTime / 1000) * 1000;

          const url = `${OKX_API_URL}?instId=${pair}&bar=${TIMEFRAME}&limit=${MAX_CANDLES}&before=${beforeTimestamp}`;
          console.log(`🔄 Attempt ${attempts + 1}/${RETRY_LIMIT} - Fetching URL: ${url}`);

          // ✅ Apply a timeout to avoid infinite wait
          const response = await Promise.race([
              fetch(url),
              new Promise((_, reject) => setTimeout(() => reject(new Error("⏳ API Timeout!")), 5000)) // 5 sec timeout
          ]) as Response;

          if (!response.ok) {
              console.warn(`⚠️ API responded with status: ${response.status}`);
              return candles;
          }

          const data = await response.json() as { data?: any[] };
          console.log(`📩 Raw API response for ${pair}:`, JSON.stringify(data).slice(0, 500)); // Log first 500 chars

          if (!data.data || data.data.length === 0) {
              console.warn(`⚠️ No candles found for ${pair}`);
              return candles;
          }

          let oldestTimestamp = startTime;

          for (const candle of data.data) {
              if (candle.length < 9) {
                  console.warn(`⚠️ Skipping malformed candle for ${pair}:`, JSON.stringify(candle));
                  continue;
              }

              const timestamp = parseInt(candle[0]);
              if (timestamp > lastTimestamp) {
                  candles.push({
                      timestamp: new Date(timestamp).toISOString(),
                      pair,
                      open: parseFloat(candle[1]),
                      high: parseFloat(candle[2]),
                      low: parseFloat(candle[3]),
                      close: parseFloat(candle[4]),
                      volume: parseFloat(candle[5]),
                      quote_volume: parseFloat(candle[6]),
                      taker_buy_base: parseFloat(candle[7]),
                      taker_buy_quote: parseFloat(candle[8]),
                  });

                  oldestTimestamp = Math.min(oldestTimestamp, timestamp);
              }
          }

          console.log(`✅ ${candles.length} candles fetched for ${pair}`);

          // ✅ Update `startTime` to fetch older candles in next request
          startTime = oldestTimestamp - 1;

          // ✅ Stop fetching if there are no more older candles
          if (candles.length < MAX_CANDLES) break;
      } catch (error) {
          attempts++;
          console.warn(`⚠️ Retry ${attempts}/${RETRY_LIMIT} for ${pair}:`, error);
      }
  }

  return candles;
}


// ✅ Store candles in Supabase
async function storeCandles(pair: string, candles: any[]) {
    try {
        const { error } = await supabase.from("candles_1D").insert(candles);

        if (error) {
            console.error(`❌ Failed to insert candles for ${pair}:`, error);
        } else {
            console.log(`✅ Inserted ${candles.length} candles for ${pair}`);
        }
    } catch (error) {
        console.error(`❌ Unexpected error inserting candles for ${pair}:`, error);
    }
}

// ✅ Sync data for a specific pair
async function syncPairData(pair: string): Promise<number> {
    const lastTimestamp = await getLastStoredTimestamp(pair);
    const newCandles = await fetchCandles(pair, lastTimestamp);

    if (newCandles.length > 0) {
        await storeCandles(pair, newCandles);
        return newCandles.length;
    }
    return 0;
}

// ✅ Send daily report email
async function sendEmailReport(newRecords: number, failedPairs: string[], errorMessage: string = "") {
    try {
        await fetch(`${SUPABASE_URL}/functions/v1/send-email`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
            },
            body: JSON.stringify({
                to: EMAIL_RECIPIENT,
                subject: "📊 Daily Market Data Sync Report",
                text: `
                    ⏳ Sync Time: ${new Date().toISOString()}
                    ✅ Inserted Candles: ${newRecords}
                    ❌ Failed Pairs: ${failedPairs.length}
                    ${failedPairs.length ? `🔴 Failed Pairs:\n${failedPairs.join(", ")}` : ""}
                    ${errorMessage ? `⚠️ Errors: ${errorMessage}` : ""}
                `,
            }),
        });
    } catch (error) {
        console.error("❌ Error sending email report:", error);
    }
}
