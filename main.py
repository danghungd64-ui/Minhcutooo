from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="Tai Xiu Prediction API - Cau Analysis",
    description="API dự đoán Tài Xỉu theo session ID thực tế. Tự động chạy cầu từ phiên cuối đến target_session.",
    version="3.0.0"
)

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
class SessionData(BaseModel):
    session: int = Field(..., description="Mã phiên (VD: 7056495)")
    dice: List[int] = Field(..., min_items=3, max_items=3, description="Mảng 3 xúc xắc")
    total: Optional[int] = None
    result: Optional[str] = None

class PredictRequest(BaseModel):
    history: List[SessionData]
    target_session: Optional[int] = None

class BridgeStep(BaseModel):
    session: int
    prediction: str
    confidence: float
    bridge: str

class PredictResponse(BaseModel):
    target_session: int
    prediction: str
    confidence: float
    detected_bridge: str
    analysis: dict
    bridge_forecast: Optional[List[BridgeStep]] = None
    forecast_until: Optional[int] = None
    gap: Optional[int] = None

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def preprocess_history(history: List[SessionData]) -> List[dict]:
    processed = []
    for item in history:
        dice_sum = sum(item.dice) if item.dice else item.total
        if dice_sum is None:
            continue
        res = item.result.lower() if item.result else ("tai" if dice_sum >= 11 else "xiu")
        processed.append({
            "session": item.session,
            "dice": item.dice,
            "total": dice_sum,
            "result": res
        })
    return processed


def detect_streak(results: List[str]) -> int:
    if not results:
        return 0
    last = results[-1]
    streak = 1
    for i in range(len(results) - 2, -1, -1):
        if results[i] == last:
            streak += 1
        else:
            break
    return streak


def detect_alternating(results: List[str]) -> int:
    if len(results) < 2:
        return 0
    count = 1
    for i in range(len(results) - 2, -1, -1):
        if results[i] != results[i + 1]:
            count += 1
        else:
            break
    return count


def detect_2_2_pattern(results: List[str]) -> Optional[str]:
    if len(results) < 4:
        return None
    last4 = results[-4:]
    if last4 == ["tai", "tai", "xiu", "xiu"]:
        return "tai"
    if last4 == ["xiu", "xiu", "tai", "tai"]:
        return "xiu"
    return None


def detect_3_3_pattern(results: List[str]) -> Optional[str]:
    if len(results) < 6:
        return None
    last6 = results[-6:]
    if last6 == ["tai", "tai", "tai", "xiu", "xiu", "xiu"]:
        return "tai"
    if last6 == ["xiu", "xiu", "xiu", "tai", "tai", "tai"]:
        return "xiu"
    return None


def analyze_bridge_pattern(results: List[str], totals: List[int]) -> tuple:
    n = len(results)
    if n == 0:
        return "tai", "Chưa đủ dữ liệu", 0.50, "Cần tối thiểu dữ liệu lịch sử."

    last_res = results[-1]
    opposite_res = "xiu" if last_res == "tai" else "tai"
    last_total = totals[-1]

    streak = detect_streak(results)
    alt = detect_alternating(results)

    p3 = detect_3_3_pattern(results)
    if p3:
        return p3, "Cầu 3-3", 0.70, f"Mẫu 3-3 hoàn tất, dự đoán {p3.upper()}."

    p2 = detect_2_2_pattern(results)
    if p2:
        return p2, "Cầu 2-2", 0.66, f"Mẫu 2-2 hoàn tất, dự đoán {p2.upper()}."

    if streak >= 4:
        return last_res, "Cầu Bệt", 0.72, f"Đang bệt {last_res.upper()} {streak} phiên liên tiếp."

    if alt >= 4:
        return opposite_res, "Cầu 1-1", 0.68, f"Đan xen {alt} phiên, đổi sang {opposite_res.upper()}."

    if last_total >= 15:
        return "xiu", "Cầu Đảo (Điểm Cực Cao)", 0.74, f"Điểm {last_total} cực cao, khả năng bẻ XỈU."
    if last_total <= 6:
        return "tai", "Cầu Đảo (Điểm Cực Thấp)", 0.74, f"Điểm {last_total} cực thấp, khả năng bẻ TÀI."

    if last_total >= 11:
        return "xiu", "Cầu Đảo Nhẹ", 0.58, f"Điểm {last_total} vùng TÀI, nghiêng về XỈU."
    return "tai", "Cầu Đảo Nhẹ", 0.58, f"Điểm {last_total} vùng XỈU, nghiêng về TÀI."


def simulate_next_totals(pred: str) -> int:
    """Tổng điểm giả lập dựa trên dự đoán (Tài ≥ 11, Xỉu ≤ 10)."""
    return 12 if pred == "tai" else 9


def forecast_bridge_until(results: List[str], totals: List[int],
                          last_session: int, target_session: int) -> List[dict]:
    """
    Chạy cầu từ last_session+1 → target_session theo ĐÚNG session ID.
    Không giới hạn bước — hỗ trợ khoảng cách lớn (VD: 7056496 → 7056500).
    """
    steps = []
    if target_session <= last_session:
        return steps

    gap = target_session - last_session

    # Giới hạn an toàn: nếu gap > 500 → chỉ mô phỏng 500 bước cuối để tránh treo API
    MAX_GAP = 500
    if gap > MAX_GAP:
        start = target_session - MAX_GAP
    else:
        start = last_session + 1

    sim_results = list(results)
    sim_totals = list(totals)

    for session in range(start, target_session + 1):
        pred, bridge, conf, _ = analyze_bridge_pattern(sim_results, sim_totals)
        steps.append({
            "session": session,
            "prediction": pred,
            "confidence": round(conf, 4),
            "bridge": bridge
        })
        sim_results.append(pred)
        sim_totals.append(simulate_next_totals(pred))

    return steps


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Tai Xiu Prediction API",
        "version": "3.0.0",
        "features": ["predict", "bridge_forecast", "session_follow"],
        "docs_url": "/docs"
    }


@app.post("/predict", response_model=PredictResponse)
def predict_tai_xiu(data: PredictRequest):
    if not data.history:
        raise HTTPException(status_code=400, detail="Mảng 'history' không được để trống.")

    sorted_history = sorted(data.history, key=lambda x: x.session)
    processed = preprocess_history(sorted_history)

    if not processed:
        raise HTTPException(status_code=400, detail="Không có dữ liệu hợp lệ sau khi xử lý.")

    results = [item["result"] for item in processed]
    totals = [item["total"] for item in processed]

    last_session = processed[-1]["session"]

    # Nếu không truyền target → mặc định phiên kế tiếp (last + 1)
    target = data.target_session if data.target_session else last_session + 1

    # Validate target > last
    if target <= last_session:
        raise HTTPException(
            status_code=400,
            detail=f"'target_session' ({target}) phải lớn hơn phiên cuối trong history ({last_session})."
        )

    gap = target - last_session

    # Dự đoán phiên kế tiếp (last_session + 1)
    prediction, bridge, confidence, note = analyze_bridge_pattern(results, totals)

    # Nếu target > last + 1 → mô phỏng chuỗi cầu đến target
    forecast = None
    if gap > 1:
        forecast = forecast_bridge_until(results, totals, last_session, target)
        # Nếu target nằm trong chuỗi forecast → lấy đúng phiên đó làm kết quả chính
        for step in forecast:
            if step["session"] == target:
                prediction = step["prediction"]
                confidence = step["confidence"]
                bridge = step["bridge"]
                note = f"Dự đoán cho phiên {target} (cách {gap} phiên từ phiên cuối)."
                break

    return {
        "target_session": target,
        "prediction": prediction,
        "confidence": confidence,
        "detected_bridge": bridge,
        "analysis": {
            "last_session": last_session,
            "last_total": totals[-1],
            "last_result": results[-1],
            "total_analyzed_sessions": len(processed),
            "gap_from_last": gap,
            "note": note
        },
        "bridge_forecast": forecast,
        "forecast_until": target if forecast else None,
        "gap": gap
    }


@app.post("/forecast")
def forecast_only(data: PredictRequest):
    """
    Chạy mô phỏng cầu theo đúng session ID đến target_session.
    """
    if not data.history:
        raise HTTPException(status_code=400, detail="Mảng 'history' không được để trống.")
    if not data.target_session:
        raise HTTPException(status_code=400, detail="Cần truyền 'target_session'.")

    sorted_history = sorted(data.history, key=lambda x: x.session)
    processed = preprocess_history(sorted_history)

    if not processed:
        raise HTTPException(status_code=400, detail="Không có dữ liệu hợp lệ.")

    results = [item["result"] for item in processed]
    totals = [item["total"] for item in processed]
    last_session = processed[-1]["session"]

    if data.target_session <= last_session:
        raise HTTPException(
            status_code=400,
            detail=f"'target_session' phải lớn hơn phiên cuối ({last_session})."
        )

    steps = forecast_bridge_until(results, totals, last_session, data.target_session)

    return {
        "from_session": last_session,
        "to_session": data.target_session,
        "gap": data.target_session - last_session,
        "total_steps": len(steps),
        "bridge_forecast": steps
    }
