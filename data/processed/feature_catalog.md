# Feature Catalog

**Total features:** 180
**Rows:** 97,584

## Feature Groups

### Numerical (scaled) (43 features)

- `item_count`
- `monthly_income`
- `household_size`
- `session_duration_min`
- `n_events`
- `n_product_views`
- `n_searches`
- `n_add_to_cart`
- `total_cart_qty`
- `avg_scroll`
- `avg_page_duration`
- `n_categories`
- `payment_installments`
- `review_score`
- `n_payments`
- `geo_lat`
- `geo_lng`
- `customer_urban`
- `cust_seller_distance_km`
- `regional_price_index`
- `seller_order_count`
- `seller_avg_price`
- `seller_item_count`
- `seller_n_categories`
- `seller_lat`
- `seller_lng`
- `customer_age_days`
- `recency_days`
- `frequency`
- `purchase_month`
- `purchase_quarter`
- `purchase_day_of_month`
- `customer_order_count`
- `cart_conversion`
- `engagement_rate`
- `search_intensity`
- `items_per_minute`
- `loyalty_numeric`
- `income_x_loyalty`
- `log_income`
- `purchase_hour`
- `purchase_dow`
- `age`

### Boolean (2 features)

- `is_logged_in`
- `is_marketing_opt_in`

### One-hot encoded (131 features)

- `device_type_desktop`
- `device_type_mobile`
- `device_type_tablet`
- `preferred_device_desktop`
- `preferred_device_mobile`
- `preferred_device_tablet`
- `browser_chrome`
- `browser_edge`
- `browser_firefox`
- `browser_other`
- `browser_safari`
- `operating_system_android`
- `operating_system_ios`
- `operating_system_linux`
- `operating_system_macos`
- `operating_system_windows`
- `traffic_source_direct`
- `traffic_source_email`
- `traffic_source_facebook`
- `traffic_source_google`
- `traffic_source_instagram`
- `traffic_medium_cpc`
- `traffic_medium_direct`
- `traffic_medium_email`
- `traffic_medium_organic`
- `traffic_medium_social`
- `landing_page_/`
- `landing_page_/category`
- `landing_page_/product`
- `landing_page_/search`
- `ip_country_AR`
- `ip_country_BR`
- `ip_country_PT`
- `ip_country_US`
- `ip_country_other`
- `gender_female`
- `gender_male`
- `gender_other`
- `marital_status_divorced`
- `marital_status_married`
- `marital_status_single`
- `marital_status_widowed`
- `education_level_bachelor`
- `education_level_high_school`
- `education_level_master`
- `education_level_none`
- `education_level_phd`
- `loyalty_tier_bronze`
- `loyalty_tier_gold`
- `loyalty_tier_platinum`
- `loyalty_tier_silver`
- `registration_channel_email`
- `registration_channel_organic`
- `registration_channel_paid_ads`
- `registration_channel_referral`
- `registration_channel_social`
- `payment_type_boleto`
- `payment_type_credit_card`
- `payment_type_debit_card`
- `payment_type_voucher`
- `primary_category_agro_industry_and_commerce`
- `primary_category_air_conditioning`
- `primary_category_art`
- `primary_category_arts_and_craftmanship`
- `primary_category_audio`
- `primary_category_auto`
- `primary_category_baby`
- `primary_category_bed_bath_table`
- `primary_category_books_general_interest`
- `primary_category_books_imported`
- `primary_category_books_technical`
- `primary_category_cds_dvds_musicals`
- `primary_category_christmas_supplies`
- `primary_category_cine_photo`
- `primary_category_computers`
- `primary_category_computers_accessories`
- `primary_category_consoles_games`
- `primary_category_construction_tools_construction`
- `primary_category_construction_tools_lights`
- `primary_category_construction_tools_safety`
- `primary_category_cool_stuff`
- `primary_category_costruction_tools_garden`
- `primary_category_costruction_tools_tools`
- `primary_category_diapers_and_hygiene`
- `primary_category_drinks`
- `primary_category_dvds_blu_ray`
- `primary_category_electronics`
- `primary_category_fashio_female_clothing`
- `primary_category_fashion_bags_accessories`
- `primary_category_fashion_childrens_clothes`
- `primary_category_fashion_male_clothing`
- `primary_category_fashion_shoes`
- `primary_category_fashion_sport`
- `primary_category_fashion_underwear_beach`
- `primary_category_fixed_telephony`
- `primary_category_flowers`
- `primary_category_food`
- `primary_category_food_drink`
- `primary_category_furniture_bedroom`
- `primary_category_furniture_decor`
- `primary_category_furniture_living_room`
- `primary_category_furniture_mattress_and_upholstery`
- `primary_category_garden_tools`
- `primary_category_health_beauty`
- `primary_category_home_appliances`
- `primary_category_home_appliances_2`
- `primary_category_home_comfort_2`
- `primary_category_home_confort`
- `primary_category_home_construction`
- `primary_category_housewares`
- `primary_category_industry_commerce_and_business`
- `primary_category_kitchen_dining_laundry_garden_furniture`
- `primary_category_la_cuisine`
- `primary_category_luggage_accessories`
- `primary_category_market_place`
- `primary_category_music`
- `primary_category_musical_instruments`
- `primary_category_office_furniture`
- `primary_category_party_supplies`
- `primary_category_perfumery`
- `primary_category_pet_shop`
- `primary_category_security_and_services`
- `primary_category_signaling_and_security`
- `primary_category_small_appliances`
- `primary_category_small_appliances_home_oven_and_coffee`
- `primary_category_sports_leisure`
- `primary_category_stationery`
- `primary_category_tablets_printing_image`
- `primary_category_telephony`
- `primary_category_toys`
- `primary_category_watches_gifts`

### Target encoded (3 features)

- `ip_region_te`
- `campaign_name_te`
- `seller_state_te`

### Derived (12 features)

- `customer_order_count`
- `cart_conversion`
- `engagement_rate`
- `search_intensity`
- `items_per_minute`
- `loyalty_numeric`
- `income_x_loyalty`
- `log_income`
- `purchase_hour`
- `purchase_dow`
- `age`
- `is_weekend`

## Excluded Features

- `avg_item_price` — r=0.92 with target (leakage: mechanically derived from order contents)
- `order_value` — target variable
- `freight_value` — component of target
- Post-purchase timestamps — not available at prediction time

## Preprocessing Artifacts

- `preprocessor.joblib` — fitted StandardScaler + target encoding maps
- Load with `joblib.load('preprocessor.joblib')` for inference