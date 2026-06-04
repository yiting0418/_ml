import onnx

print("讀取模型中...")
# 讀取剛剛匯出的模型
model = onnx.load("model.onnx")
print(f"原本的 IR Version: {model.ir_version}")

# 強制塗改版本號碼為 8
model.ir_version = 8

# 重新存檔 (這會順便把你原本分裂成兩個的 model.onnx 和 model.onnx.data 完美合併成一個！)
onnx.save(model, "model.onnx")

print("✅ 成功將模型降級為 IR Version 8！")