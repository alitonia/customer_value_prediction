# Data Dictionary - Dự án Customer Order Value Prediction

**Dự án**: Predicting Customer Order Value for an E-Commerce Platform  
**Nhóm**: G13
**Cập nhật lần cuối**: 11/07/2026

---

## 1. Tổng quan các bảng dữ liệu

| Bảng | Mô tả | Nguồn | Trạng thái |
|------|-------|-------|------------|
| `olist_orders_dataset` | Thông tin đơn hàng gốc từ Olist | Raw | Đã làm sạch |
| `olist_customers_dataset` | Thông tin khách hàng | Raw | Đã merge vào cleaned_orders |
| `olist_order_payments_dataset` | Thông tin thanh toán | Raw | Đã merge vào cleaned_orders |
| `olist_order_items_dataset` | Chi tiết sản phẩm trong đơn | Raw | Dùng để tính order_value |
| `behavioral_sessions` | Dữ liệu hành vi phiên duyệt web (synthetic) | Synthetic (Task 1.2) | Đã làm sạch |
| `cleaned_orders` | Bảng đơn hàng đã xử lý | Processed | Sẵn sàng dùng |
| `cleaned_behavioral_sessions` | Bảng hành vi đã xử lý | Processed | Sẵn sàng dùng |
| `final_dataset` | Bảng dữ liệu cuối cùng dùng cho Modeling | Merged | Sẵn sàng cho ML |

---

## 2. Mô tả chi tiết các bảng chính

### 2.1. cleaned_orders
Bảng chứa thông tin đơn hàng đã được làm sạch và bổ sung thông tin khách hàng + thanh toán.

| Cột | Kiểu dữ liệu | Mô tả | Nguồn gốc |
|-----|--------------|-------|-----------|
| order_id | string | Mã đơn hàng | olist_orders_dataset |
| customer_id | string | Mã khách hàng | olist_orders_dataset |
| order_status | string | Trạng thái đơn hàng | olist_orders_dataset |
| order_purchase_timestamp | datetime | Thời điểm đặt hàng | olist_orders_dataset |
| order_delivered_customer_date | datetime | Thời điểm giao hàng | olist_orders_dataset |
| delivery_days | int | Số ngày giao hàng | Tính toán |
| customer_city | string | Thành phố khách hàng | olist_customers_dataset |
| customer_state | string | Bang khách hàng | olist_customers_dataset |
| order_value | float | Tổng giá trị đơn hàng (BRL) | Tính từ order_items |
| payment_value | float | Tổng giá trị thanh toán | olist_order_payments_dataset |

---

### 2.2. cleaned_behavioral_sessions
Bảng dữ liệu hành vi phiên duyệt web được sinh theo **Task 1.2** (map 1-1 với order_id thật).

| Cột | Kiểu dữ liệu | Mô tả | Ghi chú |
|-----|--------------|-------|---------|
| order_id | string | Mã đơn hàng (khớp với Olist) | Khóa chính |
| session_id | string | Mã phiên duyệt web | UUID |
| device_type | string | Thiết bị sử dụng | mobile / desktop / tablet |
| referral_channel | string | Kênh traffic | search_organic, direct, social... |
| session_duration_seconds | int | Thời lượng phiên (giây) | 10 - 3600 |
| pages_viewed | int | Số trang đã xem | - |
| cart_additions | int | Số lần thêm vào giỏ hàng | >= số item thực tế |
| coupon_applied | int | Có áp dụng coupon không | 0 / 1 |
| discount_amount_pct | int | Mức giảm giá (%) | 0, 5, 10, 15, 20 |

---

## 3. Quy tắc chất lượng dữ liệu

- Chỉ sử dụng các đơn hàng có trạng thái `delivered`.
- `cart_additions` luôn lớn hơn hoặc bằng số lượng sản phẩm thực tế trong đơn.
- Không có giá trị null ở các cột quan trọng sau khi cleaning.
- `order_value` được tính bằng tổng `price` của các item trong đơn.

---

**Lưu ý cho nhóm**:
- File này dùng để tra cứu nhanh cấu trúc dữ liệu.
- Chi tiết hơn về bảng `final_dataset` xem tại file `final_dataset_dictionary.md`.