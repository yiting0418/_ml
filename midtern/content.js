// === 拓元售票系統精準 ID 設定 ===
const TIXCRAFT_IMG_ID = "#TicketForm_verifyCode-image"; 
const TIXCRAFT_INPUT_ID = "#TicketForm_verifyCode";
const CHARACTERS = "0123456789abcdefghijklmnopqrstuvwxyz";

let session = null;

// 1. 初始化模型載入 (只載入一次)
async function initModel() {
    if (session) return;
    try {
        // 🚨 告訴 ONNX 去哪裡找剛下載的 wasm 引擎
        ort.env.wasm.wasmPaths = chrome.runtime.getURL("");
        
        // 🚨 徹底關閉進階功能，強迫它只使用最基礎的 ort-wasm.wasm
        ort.env.wasm.numThreads = 1; 
        ort.env.wasm.simd = false;
        
        const modelUrl = chrome.runtime.getURL("model.onnx");
        session = await ort.InferenceSession.create(modelUrl, { 
            executionProviders: ['wasm'] 
        });
        
        console.log("✅ ONNX 模型載入成功 (使用本機基礎 WASM)！");
    } catch (e) {
        console.error("❌ 模型載入失敗:", e);
    }
}

// 2. 影像前處理 (模仿 PyTorch 的 transforms)
function preprocessImage(imgElement) {
    const canvas = document.createElement('canvas');
    canvas.width = 140;
    canvas.height = 40;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0, 140, 40);
    
    const imgData = ctx.getImageData(0, 0, 140, 40).data;
    const float32Data = new Float32Array(3 * 40 * 140);
    
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < 40 * 140; i++) {
        float32Data[i] = ((imgData[i * 4] / 255.0) - mean[0]) / std[0];
        float32Data[40 * 140 + i] = ((imgData[i * 4 + 1] / 255.0) - mean[1]) / std[1];
        float32Data[2 * 40 * 140 + i] = ((imgData[i * 4 + 2] / 255.0) - mean[2]) / std[2];
    }

    return new ort.Tensor('float32', float32Data, [1, 3, 40, 140]);
}

// 3. 找出陣列中最大值的索引 (Argmax)
function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

// 4. 核心解題邏輯
async function solve() {
    console.log("🕵️‍♂️ [Tixcraft 外掛] 已喚醒！正在掃描網頁元素...");
    
    const img = document.querySelector(TIXCRAFT_IMG_ID);
    const input = document.querySelector(TIXCRAFT_INPUT_ID);

    if (!img) {
        console.log("❌ 找不到圖片，目前的設定是:", TIXCRAFT_IMG_ID);
        return;
    }
    if (!input) {
        console.log("❌ 找不到輸入框，目前的設定是:", TIXCRAFT_INPUT_ID);
        return;
    }

    if (!img.complete || img.naturalWidth === 0) {
        console.log("⏳ 圖片還在載入中，稍等...");
        img.onload = solve;
        return;
    }

    await initModel();

    // 🚨 加入這段：如果模型載入失敗，就立刻煞車，避免產生 null 錯誤
    if (!session) {
        console.log("⏸️ 大腦尚未準備好，停止推論。");
        return;
    }


    try {
        console.log("🚀 開始瀏覽器端推論...");
        const tensor = preprocessImage(img);
        
        const feeds = { [session.inputNames[0]]: tensor };
        const results = await session.run(feeds);
        
        const p1 = argMax(Array.from(results.out1.data));
        const p2 = argMax(Array.from(results.out2.data));
        const p3 = argMax(Array.from(results.out3.data));
        const p4 = argMax(Array.from(results.out4.data));
        
        const prediction = CHARACTERS[p1] + CHARACTERS[p2] + CHARACTERS[p3] + CHARACTERS[p4];
        console.log("🎯 辨識結果:", prediction);
        
        // 自動填入
        input.value = prediction;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));

    } catch (e) {
        console.error("❌ 推論發生錯誤:", e);
    }
}

// 監聽網頁變化 (應付動態重整，或是點擊圖片換一張的功能)
const observer = new MutationObserver(() => {
    // 只有在看到圖片，而且輸入框是空的時候才觸發，避免無限迴圈
    if (document.querySelector(TIXCRAFT_IMG_ID) && !document.querySelector(TIXCRAFT_INPUT_ID).value) {
        solve();
    }
});

observer.observe(document.body, { childList: true, subtree: true });

// 初始啟動
solve();