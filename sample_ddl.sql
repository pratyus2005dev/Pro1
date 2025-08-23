-- Sample DDL for Enhanced Schema Mapping System
-- This demonstrates the schema structure for source and target systems

-- SOURCE SYSTEM (Legacy Application)
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email_address VARCHAR(100) UNIQUE,
    phone_number VARCHAR(20),
    date_of_birth DATE,
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    total_orders INTEGER DEFAULT 0,
    lifetime_value DECIMAL(10,2) DEFAULT 0.00
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TIMESTAMP NOT NULL,
    order_status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(10,2) NOT NULL,
    shipping_address TEXT,
    billing_address TEXT,
    payment_method VARCHAR(50),
    delivery_notes TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    description TEXT,
    category VARCHAR(50),
    unit_price DECIMAL(8,2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    supplier_id INTEGER,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(8,2) NOT NULL,
    discount_amount DECIMAL(8,2) DEFAULT 0.00,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- TARGET SYSTEM (Modern Application)
CREATE TABLE clients (
    client_id INTEGER PRIMARY KEY,
    given_name VARCHAR(50) NOT NULL,
    family_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    contact_phone VARCHAR(20),
    birth_date DATE,
    member_since TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    account_status VARCHAR(20) DEFAULT 'active',
    order_count INTEGER DEFAULT 0,
    total_spent DECIMAL(10,2) DEFAULT 0.00
);

CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY,
    client_id INTEGER NOT NULL,
    transaction_date TIMESTAMP NOT NULL,
    transaction_status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(10,2) NOT NULL,
    shipping_address TEXT,
    billing_address TEXT,
    payment_type VARCHAR(50),
    delivery_instructions TEXT,
    FOREIGN KEY (client_id) REFERENCES clients(client_id)
);

CREATE TABLE inventory (
    inventory_id INTEGER PRIMARY KEY,
    item_name VARCHAR(100) NOT NULL,
    item_description TEXT,
    item_category VARCHAR(50),
    price DECIMAL(8,2) NOT NULL,
    available_quantity INTEGER DEFAULT 0,
    vendor_id INTEGER,
    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE transaction_items (
    line_item_id INTEGER PRIMARY KEY,
    transaction_id INTEGER NOT NULL,
    inventory_id INTEGER NOT NULL,
    item_quantity INTEGER NOT NULL,
    item_price DECIMAL(8,2) NOT NULL,
    discount DECIMAL(8,2) DEFAULT 0.00,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id),
    FOREIGN KEY (inventory_id) REFERENCES inventory(inventory_id)
);

-- Additional metadata for enhanced mapping
-- This information helps the system understand relationships and constraints

-- Primary Key Information
-- customers.customer_id -> clients.client_id (PK to PK mapping)
-- orders.order_id -> transactions.transaction_id (PK to PK mapping)
-- products.product_id -> inventory.inventory_id (PK to PK mapping)

-- Foreign Key Relationships
-- orders.customer_id -> customers.customer_id
-- transactions.client_id -> clients.client_id
-- order_items.order_id -> orders.order_id
-- transaction_items.transaction_id -> transactions.transaction_id

-- Business Logic Mappings
-- customer_id -> client_id (same entity, different naming)
-- first_name -> given_name (same concept, different naming)
-- last_name -> family_name (same concept, different naming)
-- email_address -> email (same concept, different naming)
-- order_date -> transaction_date (same concept, different naming)
-- total_amount -> total_amount (same concept, same naming)
-- product_name -> item_name (same concept, different naming)
-- unit_price -> price (same concept, different naming)