FROM python:3.11-slim

WORKDIR /app

# 1. Cài uv qua pip như bạn đã làm thành công trước đó
RUN pip install uv

# 2. Copy file cấu hình
COPY pyproject.toml ./

# 3. Cài đặt TOÀN BỘ dependencies từ pyproject.toml
# Lệnh này sẽ cài tất cả mọi thứ bạn khai báo trong file toml
RUN uv pip install --system .

# 4. Nếu có thư viện nào bạn chưa kịp khai báo trong toml mà muốn cài thêm
# thì mới dùng lệnh dưới đây (nhưng tốt nhất là nên cho hết vào toml)
RUN uv pip install --system prometheus_client

# 5. Copy toàn bộ source code (bao gồm thư mục src)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]