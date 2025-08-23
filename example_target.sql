-- Target Database Schema (Modern System)

CREATE TABLE client_profiles (
    client_id BIGINT PRIMARY KEY,
    given_name VARCHAR(60) NOT NULL,
    family_name VARCHAR(120) NOT NULL,
    email_addr VARCHAR(300) UNIQUE,
    mobile_phone VARCHAR(25),
    birth_date DATE,
    registration_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active_status BOOLEAN DEFAULT TRUE,
    client_tier VARCHAR(25) DEFAULT 'standard',
    loyalty_points INT DEFAULT 0
);

CREATE TABLE item_catalog (
    item_id BIGINT PRIMARY KEY,
    item_title VARCHAR(250) NOT NULL,
    item_details TEXT,
    selling_price DECIMAL(15,4) NOT NULL,
    inventory_count INT DEFAULT 0,
    category_ref INT,
    manufacturer VARCHAR(120),
    discontinued_flag BOOLEAN DEFAULT FALSE,
    creation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    weight_kg DECIMAL(8,3),
    dimensions VARCHAR(50)
);

CREATE TABLE purchase_orders (
    purchase_id BIGINT PRIMARY KEY,
    client_ref BIGINT NOT NULL,
    purchase_datetime TIMESTAMP NOT NULL,
    gross_amount DECIMAL(15,4) NOT NULL,
    current_status VARCHAR(30) DEFAULT 'submitted',
    delivery_address JSON,
    invoice_address JSON,
    payment_type VARCHAR(60),
    discount_value DECIMAL(12,4) DEFAULT 0.0000,
    tax_value DECIMAL(12,4) DEFAULT 0.0000,
    special_instructions TEXT,
    tracking_number VARCHAR(100)
);

CREATE TABLE order_line_items (
    line_item_id BIGINT PRIMARY KEY,
    purchase_ref BIGINT NOT NULL,
    item_ref BIGINT NOT NULL,
    ordered_quantity INT NOT NULL,
    item_price DECIMAL(15,4) NOT NULL,
    subtotal DECIMAL(15,4) NOT NULL,
    discount_rate DECIMAL(7,4) DEFAULT 0.0000,
    tax_rate DECIMAL(7,4) DEFAULT 0.0000
);

CREATE TABLE product_categories (
    cat_id BIGINT PRIMARY KEY,
    cat_name VARCHAR(120) NOT NULL,
    parent_cat_id BIGINT,
    cat_description TEXT,
    active_flag BOOLEAN DEFAULT TRUE,
    sort_order INT DEFAULT 0,
    icon_url VARCHAR(500)
);

CREATE TABLE system_users (
    sys_user_id BIGINT PRIMARY KEY,
    login_name VARCHAR(60) UNIQUE NOT NULL,
    pwd_hash VARCHAR(300) NOT NULL,
    contact_email VARCHAR(300) UNIQUE NOT NULL,
    display_name VARCHAR(180),
    user_role VARCHAR(40) DEFAULT 'end_user',
    last_access TIMESTAMP,
    profile_status VARCHAR(25) DEFAULT 'enabled',
    creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferences JSON
);

CREATE TABLE inventory_movements (
    movement_id BIGINT PRIMARY KEY,
    item_ref BIGINT NOT NULL,
    movement_type VARCHAR(20) NOT NULL,
    quantity_change INT NOT NULL,
    movement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reference_doc VARCHAR(100),
    notes TEXT
);

CREATE TABLE customer_addresses (
    address_id BIGINT PRIMARY KEY,
    client_ref BIGINT NOT NULL,
    address_type VARCHAR(20) DEFAULT 'shipping',
    street_address VARCHAR(200),
    city_name VARCHAR(100),
    state_province VARCHAR(100),
    postal_code VARCHAR(20),
    country_code VARCHAR(5) DEFAULT 'US',
    is_primary BOOLEAN DEFAULT FALSE
);