# Final Dataset Dictionary

**Mô tả**: Bảng dữ liệu cuối cùng được sử dụng cho bài toán dự đoán giá trị đơn hàng (`order_value`).  
**Mục đích**: Cung cấp cho ML team các feature đã được xử lý và engineered.  
**Số dòng**: ~96,000 (chỉ các order delivered)  
**Target**: `order_value` / `log_order_value`

---

## Danh sách các cột

| Tên cột                    | Kiểu dữ liệu | Mô tả                                                  | Loại     | Ghi chú cho ML |
|---------------------------|--------------|--------------------------------------------------------|----------|----------------|
| order_id                  | string       | Mã đơn hàng                                            | ID       | - |
| session_id                | string       | Mã phiên duyệt web                                     | ID       | - |
| device_type               | category     | Thiết bị sử dụng (mobile/desktop/tablet)               | Feature  | Nên encode   |
| referral_channel          | category     | Kênh traffic                                           | Feature  | Nên encode   |
| session_duration_seconds  | int          | Thời lượng phiên (giây)                                | Feature  | - |
| pages_viewed              | int          | Số trang đã xem                                        | Feature  | - |
| cart_additions            | int          | Số lần thêm vào giỏ hàng                               | Feature  | - |
| coupon_applied            | int          | Có dùng coupon không (0/1)                             | Feature  | - |
| discount_amount_pct       | int          | Mức giảm giá nếu có coupon                             | Feature  | - |
| order_value               | float        | Giá trị đơn hàng (BRL)                                 | Target   | Target gốc   |
| log_order_value           | float        | log1p(order_value)                                     | Target   | **Khuyến nghị dùng** |
| payment_value             | float        | Giá trị thanh toán thực tế                             | Feature  | - |
| customer_city             | category     | Thành phố khách hàng                                   | Feature  | Có thể bỏ hoặc group |
| customer_state            | category     | Bang khách hàng                                        | Feature  | - |
| order_status              | string       | Trạng thái đơn hàng                                    | -        | Chỉ delivered |
| delivery_days             | float        | Số ngày giao hàng                                      | Feature  | - |
| avg_time_per_page         | float        | Thời gian trung bình xem 1 trang                       | Feature  | Engineered |
| cart_per_page             | float        | Tỷ lệ thêm giỏ / số trang xem                          | Feature  | Engineered |
| is_high_engagement        | int          | Phiên có mức độ tương tác cao (0/1)                    | Feature  | Engineered |
| has_coupon                | int          | Có dùng coupon (0/1)                                   | Feature  | = coupon_applied |

---

## Các feature được tạo mới (Engineered Features)

| Feature                | Công thức                                      | Ý nghĩa |
|------------------------|------------------------------------------------|--------|
| `log_order_value`      | `log1p(order_value)`                           | Giảm độ lệch phải của target |
| `avg_time_per_page`    | `session_duration_seconds / pages_viewed`      | Mức độ quan tâm của khách hàng |
| `cart_per_page`        | `cart_additions / pages_viewed`                | Hiệu quả chuyển đổi hành vi |
| `is_high_engagement`   | `duration > median AND pages_viewed > median`  | Phiên tương tác cao |

---

## Khuyến nghị cho ML Team

- **Target chính**: Nên dùng `log_order_value` thay vì `order_value` vì dữ liệu bị lệch phải mạnh.
- Các feature `avg_time_per_page` và `cart_per_page` có khả năng mang tín hiệu dự đoán tốt.
- Nên thực hiện encoding cho các cột categorical (`device_type`, `referral_channel`).
- Không có data leakage ở mức feature hiện tại.

---

**Cập nhật lần cuối**: 11/07/2026