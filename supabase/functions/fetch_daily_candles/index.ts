import { createClient } from "@supabase/supabase-js";
import fetch from "node-fetch";

const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const OKX_API_URL = "https://www.okx.com/api/v5/market/candles";
const OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT";
const MAX_CANDLES = 1000; // Max allowed candles per OKX request
const TIMEFRAME = "1D"; // Daily candles
const EMAIL_RECIPIENT = "robert@rorostudio.com"; // 📩 Daily Report
const RETRY_LIMIT = 3; // Max retries per failed request

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

export default async function handler(req: any, res: any) {
    console.log("📡 Starting daily market data fetch...");

    try {
        const instruments = await fetchInstruments();
        if (!instruments.length) throw new Error("No active instruments found!");

        let totalInserted = 0;
        let failedPairs: string[] = [];

        const results = await Promise.allSettled(
            instruments.map(pair => syncPairData(pair))
        );

        results.forEach((result, index) => {
            if (result.status === "fulfilled") totalInserted += result.value;
            else failedPairs.push(instruments[index]);
        });

        await sendEmailReport(totalInserted, failedPairs);
        console.log(`✅ Sync complete: ${totalInserted} candles inserted.`);

        res.status(200).json({ success: true, inserted: totalInserted, failed: failedPairs.length });
    } catch (error) {
        console.error("🚨 Fatal Error:", error);
        res.status(500).json({ success: false, error: error.message });
    }
}

// 🟢 Fetch all active USDT trading pairs
async function fetchInstruments(): Promise<string[]> {
    console.log("📊 Fetching available USDT spot trading pairs...");
    const response = await fetch(OKX_INSTRUMENTS_URL);
    const data = await response.json();

    return data.data
        .filter((x: any) => x.instId.endsWith("-USDT") && x.state === "live")
        .map((x: any) => x.instId);
}

// 🔍 Get last stored timestamp to fetch only missing candles
async function getLastStoredTimestamp(pair: string): Promise<number> {
    const { data, error } = await supabase
        .from("candles_1D")
        .select("timestamp")
        .eq("pair", pair)
        .order("timestamp", { ascending: false })
        .limit(1);

    if (error) {
        console.error(`⚠️ Error fetching last timestamp for ${pair}:`, error);
        return 0; // Default to fetching from the beginning
    }
    return data.length ? data[0].timestamp : 0;
}

// 📥 Fetch missing candles from OKX with retries
async function fetchCandles(pair: string, lastTimestamp: number): Promise<any[]> {
    console.log(`📡 Fetching candles for ${pair} (since ${new Date(lastTimestamp).toISOString()})`);

    let candles: any[] = [];
    let attempts = 0;

    while (attempts < RETRY_LIMIT) {
        try {
            const url = `${OKX_API_URL}?instId=${pair}&bar=${TIMEFRAME}&limit=${MAX_CANDLES}`;
            const response = await fetch(url);
            const data = await response.json();

            if (!data.data || data.data.length === 0) return candles;

            for (const candle of data.data) {
                const timestamp = parseInt(candle[0]);
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
            return candles;
        } catch (error) {
            attempts++;
            console.warn(`⚠️ Retry ${attempts}/${RETRY_LIMIT} for ${pair}...`);
            await new Promise(resolve => setTimeout(resolve, 2000 * attempts));
        }
    }
    console.error(`❌ Failed to fetch ${pair} after ${RETRY_LIMIT} attempts.`);
    return [];
}

// 📤 Store candles in Supabase
async function storeCandles(pair: string, candles: any[]) {
    const { error } = await supabase.from("candles_1D").insert(candles, { upsert: true });

    if (error) console.error(`❌ Failed to insert candles for ${pair}:`, error);
    else console.log(`✅ Inserted ${candles.length} candles for ${pair}`);
}

// 🔄 Sync data for a specific pair
async function syncPairData(pair: string): Promise<number> {
    const lastTimestamp = await getLastStoredTimestamp(pair);
    const newCandles = await fetchCandles(pair, lastTimestamp);

    if (newCandles.length > 0) {
        await storeCandles(pair, newCandles);
        return newCandles.length;
    }
    return 0;
}

// 📩 Send daily report email
async function sendEmailReport(newRecords: number, failedPairs: string[]) {
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
            `,
        }),
    });
}