// ✅ Load Supabase Edge Runtime Definitions
//import "https://esm.sh/@supabase/functions-js@2/edge-runtime.d.ts";
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

        let totalInserted = 0;
        let failedPairs: string[] = [];

        for (const pair of instruments) {
            try {
                const insertedCount = await syncPairData(pair);
                totalInserted += insertedCount;
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
    const response = await fetch(OKX_INSTRUMENTS_URL);
    const data = await response.json() as { data?: any[] };

    return data.data?.filter(x => (x.instId.endsWith("-USDT") || x.instId.endsWith("-USDC")) && x.state === "live").map(x => x.instId) || [];
}

// ✅ Get last stored timestamp to fetch only missing candles
async function getLastStoredTimestamp(pair: string): Promise<number> {
    const { data, error } = await supabase
        .from("candles_1D")
        .select("timestamp")
        .eq("pair", pair)
        .order("timestamp", { ascending: false })
        .limit(1);

    if (error) {
        console.error(`⚠️ Error fetching last timestamp for ${pair}:`, error);
        return 0;
    }
    return data.length ? data[0].timestamp : 0;
}

// ✅ Fetch missing candles from OKX with retries
async function fetchCandles(pair: string, lastTimestamp: number): Promise<any[]> {
    console.log(`📡 Fetching candles for ${pair} (since ${new Date(lastTimestamp).toISOString()})`);

    let candles: any[] = [];
    let startTime = lastTimestamp || 1483228800000;
    let attempts = 0;

    while (startTime < Date.now() && attempts < RETRY_LIMIT) {
        try {
            const url = `${OKX_API_URL}?instId=${pair}&bar=${TIMEFRAME}&limit=${MAX_CANDLES}&before=${startTime}`;
            const response = await fetch(url);
            const data = await response.json() as { data?: any[] };

            if (!data.data || data.data.length === 0) return candles;

            for (const candle of data.data) {
              const timestamp = new Date(parseInt(candle[0])).toISOString(); // Convert UNIX timestamp to ISO format
                if (timestamp > lastTimestamp) {
                    candles.push({
                        timestamp,
                        pair,
                        open: parseFloat(candle[1]),
                        high: parseFloat(candle[2]),
                        low: parseFloat(candle[3]),
                        close: parseFloat(candle[4]),
                        volume: parseFloat(candle[5]),
                    });
                }
            }

            startTime = candles.length ? candles[candles.length - 1].timestamp - 86400000 : Date.now();

        } catch (error) {
            attempts++;
            console.warn(`⚠️ Retry ${attempts}/${RETRY_LIMIT} for ${pair}...`);
        }
    }

    return candles;
}

// ✅ Store candles in Supabase
async function storeCandles(pair: string, candles: any[]) {
    const { error } = await supabase.from("candles_1D").insert(candles);

    if (error) console.error(`❌ Failed to insert candles for ${pair}:`, error);
    else console.log(`✅ Inserted ${candles.length} candles for ${pair}`);
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
}
