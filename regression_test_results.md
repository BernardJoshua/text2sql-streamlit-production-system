# Text-to-SQL Regression Test Report

## Overall accuracy

- Total cases: **43**
- Valid golden cases: **42**
- Schema valid rate: **69.05%**
- Execution success rate: **69.05%**
- Result accuracy: **40.48%**

## Accuracy by test type

- **join**: cases=17, schema=23.53%, execution=23.53%, result=0.00%
- **single**: cases=25, schema=100.00%, execution=100.00%, result=68.00%

## Accuracy by operation

- **avg**: cases=8, schema=87.50%, execution=87.50%, result=75.00%
- **count**: cases=8, schema=87.50%, execution=87.50%, result=50.00%
- **lookup**: cases=5, schema=40.00%, execution=40.00%, result=0.00%
- **max**: cases=8, schema=75.00%, execution=75.00%, result=37.50%
- **sum**: cases=13, schema=53.85%, execution=53.85%, result=30.77%

## Accuracy by database

- **car_retails**: cases=6, schema=83.33%, execution=83.33%, result=50.00%
- **cars**: cases=6, schema=66.67%, execution=66.67%, result=33.33%
- **regional_sales**: cases=4, schema=75.00%, execution=75.00%, result=50.00%
- **restaurant**: cases=4, schema=50.00%, execution=50.00%, result=0.00%
- **retail_complains**: cases=5, schema=80.00%, execution=80.00%, result=20.00%
- **retail_world**: cases=4, schema=50.00%, execution=50.00%, result=25.00%
- **retails**: cases=4, schema=75.00%, execution=75.00%, result=75.00%
- **sales**: cases=5, schema=60.00%, execution=60.00%, result=60.00%
- **superstore**: cases=4, schema=75.00%, execution=75.00%, result=50.00%

## Failed cases

### retail_complaints_single_count_issue
- DB: `retail_complains`
- Tables: `['events']`
- Type: `single`
- Operation: `count`
- Question: How many complaints were about Billing disputes?
- Expected SQL:
```sql
SELECT COUNT("Complaint ID") FROM events WHERE Issue = 'Billing disputes';
```
- Generated SQL:
```sql
SELECT COUNT("Complaint ID") FROM events WHERE "Submitted via" = 'Billing disputes'
```
- Schema valid: `True`
- Execution success: `True`

### retail_complaints_single_top_issue
- DB: `retail_complains`
- Tables: `['events']`
- Type: `single`
- Operation: `max`
- Question: Which complaint issue appears most often?
- Expected SQL:
```sql
SELECT Issue FROM events GROUP BY Issue ORDER BY COUNT("Complaint ID") DESC LIMIT 1;
```
- Generated SQL:
```sql
SELECT "Issue" FROM events WHERE "Sub-product" = 'Complaint' ORDER BY "Submitted via" DESC LIMIT 1
```
- Schema valid: `True`
- Execution success: `True`

### retail_complaints_join_count_city_issue
- DB: `retail_complains`
- Tables: `['client', 'events']`
- Type: `join`
- Operation: `count`
- Question: Among clients from Portland, how many complaints were about Billing disputes?
- Expected SQL:
```sql
SELECT COUNT(T2."Complaint ID") FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.city = 'Portland' AND T2.Issue = 'Billing disputes';
```
- Generated SQL:
```sql
SELECT COUNT(T2."Complaint ID") FROM client AS T1 JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.city = 'Portland' AND T2.Issue = 'Billing disputes'
```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `undefined_aliases=['T2']`

### retail_complaints_join_top_city
- DB: `retail_complains`
- Tables: `['client', 'events']`
- Type: `join`
- Operation: `max`
- Question: Which city had the most complaints?
- Expected SQL:
```sql
SELECT T1.city FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID GROUP BY T1.city ORDER BY COUNT(T2."Complaint ID") DESC LIMIT 1;
```
- Generated SQL:
```sql
SELECT city FROM client WHERE state = 'complaint' GROUP BY city ORDER BY COUNT(city) DESC LIMIT 1
```
- Schema valid: `True`
- Execution success: `True`

### car_retails_join_sum_quantity_product
- DB: `car_retails`
- Tables: `['products', 'orderdetails']`
- Type: `join`
- Operation: `sum`
- Question: Calculate the total quantity ordered for 18th Century Vintage Horse Carriage.
- Expected SQL:
```sql
SELECT SUM(T2.quantityOrdered) FROM products AS T1 INNER JOIN orderdetails AS T2 ON T1.productCode = T2.productCode WHERE T1.productName = '18th Century Vintage Horse Carriage';
```
- Generated SQL:
```sql
SELECT SUM(T2.quantityOrdered) FROM products AS T1 JOIN orderdetails AS T2 ON T1.productCode = T2.productCode WHERE T1.productName = '18th Century Vintage Horse Carriage'
```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `undefined_aliases=['T2']`

### car_retails_join_top_product_quantity
- DB: `car_retails`
- Tables: `['products', 'orderdetails']`
- Type: `join`
- Operation: `sum`
- Question: Which product was ordered the most?
- Expected SQL:
```sql
SELECT T1.productName FROM products AS T1 INNER JOIN orderdetails AS T2 ON T1.productCode = T2.productCode GROUP BY T1.productName ORDER BY SUM(T2.quantityOrdered) DESC LIMIT 1;
```
- Generated SQL:
```sql
SELECT productCode FROM orderdetails ORDER BY quantityOrdered DESC LIMIT 1
```
- Schema valid: `True`
- Execution success: `True`

### car_retails_join_top_customer_payment
- DB: `car_retails`
- Tables: `['customers', 'payments']`
- Type: `join`
- Operation: `sum`
- Question: Which customer made the highest total payment?
- Expected SQL:
```sql
SELECT T1.customerName FROM customers AS T1 INNER JOIN payments AS T2 ON T1.customerNumber = T2.customerNumber GROUP BY T1.customerName ORDER BY SUM(T2.amount) DESC LIMIT 1;
```
- Generated SQL:
```sql
SELECT customerName FROM customers ORDER BY creditLimit DESC LIMIT 1
```
- Schema valid: `True`
- Execution success: `True`

### cars_single_avg_mpg
- DB: `cars`
- Tables: `['data']`
- Type: `single`
- Operation: `avg`
- Question: What is the average mpg of all cars?
- Expected SQL:
```sql
SELECT AVG(mpg) FROM data;
```
- Generated SQL:
```sql
SELECT AVG(mpg) FROM data WHERE car_name = 'cars'
```
- Schema valid: `True`
- Execution success: `True`

### cars_join_cheapest_cylinders
- DB: `cars`
- Tables: `['data', 'price']`
- Type: `join`
- Operation: `max`
- Question: How many cylinders does the cheapest car have?
- Expected SQL:
```sql
SELECT T1.cylinders FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID ORDER BY T2.price ASC LIMIT 1;
```
- Generated SQL:
```sql
SELECT T1.cylinders FROM data AS T1 JOIN price AS T2 ON T1.ID = T2.ID ORDER BY T2.price ASC LIMIT 1
```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `undefined_aliases=['T2']`

### cars_join_country_chevy
- DB: `cars`
- Tables: `['data', 'production']`
- Type: `join`
- Operation: `lookup`
- Question: Which country does Chevy C20 come from?
- Expected SQL:
```sql
SELECT T2.country FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID WHERE lower(T1.car_name) = 'chevy c20';
```
- Generated SQL:
```sql
SELECT country FROM production WHERE model_year = 'chevy c20'
```
- Schema valid: `True`
- Execution success: `True`

### cars_join_avg_price_8_cyl
- DB: `cars`
- Tables: `['data', 'price']`
- Type: `join`
- Operation: `avg`
- Question: What is the average price of cars with 8 cylinders?
- Expected SQL:
```sql
SELECT AVG(T2.price) FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1.cylinders = 8;
```
- Generated SQL:
```sql
SELECT AVG(T2.price) FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1."cylinders" = 8
```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `undefined_aliases=['T2']`

### sales_join_top_salesperson
- DB: `sales`
- Tables: `['Sales', 'Employees']`
- Type: `join`
- Operation: `sum`
- Question: Which salesperson sold the most items in total?
- Expected SQL:
```sql
SELECT T2.FirstName, T2.LastName FROM Sales AS T1 INNER JOIN Employees AS T2 ON T1.SalesPersonID = T2.EmployeeID GROUP BY T2.FirstName, T2.LastName ORDER BY SUM(T1.Quantity) DESC LIMIT 1;
```
- Generated SQL:
```sql
SELECT T1.FirstName, T1.LastName, SUM(T2.Quantity) AS total_items FROM Employees AS T1 JOIN Sales AS T2 ON T1.EmployeeID = T2.SalesPersonID GROUP BY T1.FirstName, T1.LastName ORDER BY total_items DESC LIMIT 1
```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `undefined_aliases=['T2']`

### sales_join_top_product_quantity
- DB: `sales`
- Tables: `['Sales', 'Products']`
- Type: `join`
- Operation: `sum`
- Question: Name the product that sold the highest quantity.
- Expected SQL:
```sql
SELECT T2.Name FROM Sales AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID GROUP BY T2.Name ORDER BY SUM(T1.Quantity) DESC LIMIT 1;
```
- Generated SQL:
```sql

```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `empty_sql`
- Generated error: `No generated SQL candidate passed parse, selected-schema validation, and DuckDB EXPLAIN. This is a hard rejection, not an execution failure.`

### restaurant_single_count_japanese
- DB: `restaurant`
- Tables: `['generalinfo']`
- Type: `single`
- Operation: `count`
- Question: How many restaurants serve Japanese food?
- Expected SQL:
```sql
SELECT COUNT(id_restaurant) FROM generalinfo WHERE food_type = 'japanese';
```
- Generated SQL:
```sql
SELECT COUNT(id_restaurant) FROM generalinfo WHERE food_type = 'japan'
```
- Schema valid: `True`
- Execution success: `True`

### restaurant_single_food_type
- DB: `restaurant`
- Tables: `['generalinfo']`
- Type: `single`
- Operation: `lookup`
- Question: What food type does Plearn-Thai Cuisine serve?
- Expected SQL:
```sql
SELECT food_type FROM generalinfo WHERE lower(label) = 'plearn-thai cuisine';
```
- Generated SQL:
```sql
SELECT food_type FROM generalinfo WHERE city = 'plearn-thai cuisine'
```
- Schema valid: `True`
- Execution success: `True`

### restaurant_join_county_region
- DB: `restaurant`
- Tables: `['generalinfo', 'geographic']`
- Type: `join`
- Operation: `lookup`
- Question: What is the county and region of Plearn-Thai Cuisine restaurant?
- Expected SQL:
```sql
SELECT T2.County, T2.region FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city_lower = T2.city_lower WHERE lower(T1.label) = 'plearn-thai cuisine';
```
- Generated SQL:
```sql
SELECT T2.county, T2.region FROM generalinfo AS T1 JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.label = 'Plearn-Thai Cuisine'
```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `undefined_aliases=['T2']`

### restaurant_join_street
- DB: `restaurant`
- Tables: `['generalinfo', 'location']`
- Type: `join`
- Operation: `lookup`
- Question: What is the street name of Plearn-Thai Cuisine restaurant?
- Expected SQL:
```sql
SELECT T2.street_name FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE lower(T1.label) = 'plearn-thai cuisine';
```
- Generated SQL:
```sql
SELECT t2.street_name FROM generalinfo AS t1 INNER JOIN location AS T2 ON t1.city = T2.city WHERE T1.label = 'Plearn-Thai Cuisine restaurant'
```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `undefined_aliases=['T1', 'T2']`

### regional_sales_single_sum_sales_amount
- DB: `regional_sales`
- Tables: `['Sales Orders']`
- Type: `single`
- Operation: `sum`
- Question: What is the total sales amount?
- Expected SQL:
```sql
SELECT SUM("Order Quantity" * "Unit Price") FROM "Sales Orders";
```
- Generated SQL:
```sql
SELECT SUM("Order Quantity") FROM "Sales Orders"
```
- Schema valid: `True`
- Execution success: `True`

### regional_sales_join_top_product
- DB: `regional_sales`
- Tables: `['Sales Orders', 'Products']`
- Type: `join`
- Operation: `sum`
- Question: Which product generated the highest sales amount?
- Expected SQL:
```sql
SELECT T2."Product Name" FROM "Sales Orders" AS T1 INNER JOIN Products AS T2 ON T1."_ProductID" = T2.ProductID GROUP BY T2."Product Name" ORDER BY SUM(T1."Order Quantity" * T1."Unit Price") DESC LIMIT 1;
```
- Generated SQL:
```sql
SELECT T2."Product Name", SUM(T1."Order Quantity" * T1."Unit Price") AS sales_amount FROM "Sales Orders" AS T1 JOIN Products AS T2 ON T1._ProductID = T2.ProductID GROUP BY T2."Product Name" ORDER BY sales_amount DESC LIMIT 1
```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `undefined_aliases=['T2']`

### retail_world_single_top_unit_price
- DB: `retail_world`
- Tables: `['Products']`
- Type: `single`
- Operation: `max`
- Question: Which product has the highest unit price?
- Expected SQL:
```sql
SELECT ProductName FROM Products ORDER BY UnitPrice DESC LIMIT 1;
```
- Generated SQL:
```sql
SELECT ProductName FROM Products ORDER BY UnitsInStock + UnitsOnOrder DESC LIMIT 1
```
- Schema valid: `True`
- Execution success: `True`

### retail_world_join_beverages
- DB: `retail_world`
- Tables: `['Products', 'Categories']`
- Type: `join`
- Operation: `lookup`
- Question: What are the products that belong to the beverage category?
- Expected SQL:
```sql
SELECT T1.ProductName FROM Products AS T1 INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID WHERE T2.CategoryName = 'Beverages';
```
- Generated SQL:
```sql
SELECT T1.ProductName FROM Products AS T1 JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID WHERE T2.CategoryName = 'Beverages'
```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `undefined_aliases=['T2']`

### retail_world_join_supplier_highest_price
- DB: `retail_world`
- Tables: `['Products', 'Suppliers']`
- Type: `join`
- Operation: `max`
- Question: What is the name of the company that supplies the product with the highest unit price?
- Expected SQL:
```sql
SELECT T2.CompanyName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID ORDER BY T1.UnitPrice DESC LIMIT 1;
```
- Generated SQL:
```sql
SELECT T2.CompanyName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2 . SupplierID ORDER BY UnitsInStock + UnitsOnOrder DESC
```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `undefined_aliases=['T2']`

### superstore_single_count_west_orders
- DB: `superstore`
- Tables: `['west_superstore']`
- Type: `single`
- Operation: `count`
- Question: How many orders were placed in the West superstore?
- Expected SQL:
```sql
SELECT COUNT("Order ID") FROM west_superstore;
```
- Generated SQL:
```sql
SELECT COUNT(DISTINCT "Order ID") FROM west_superstore WHERE Region = 'West'
```
- Schema valid: `True`
- Execution success: `True`

### superstore_join_product_west
- DB: `superstore`
- Tables: `['west_superstore', 'product']`
- Type: `join`
- Operation: `sum`
- Question: Which product was ordered the most in the West superstore?
- Expected SQL:
```sql
SELECT T2."Product Name" FROM west_superstore AS T1 INNER JOIN product AS T2 ON T1."Product ID" = T2."Product ID" GROUP BY T2."Product Name" ORDER BY SUM(T1.Quantity) DESC LIMIT 1;
```
- Generated SQL:
```sql

```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `empty_sql`
- Generated error: `No generated SQL candidate passed parse, selected-schema validation, and DuckDB EXPLAIN. This is a hard rejection, not an execution failure.`

### retails_join_total_customer_orders
- DB: `retails`
- Tables: `['customer', 'orders']`
- Type: `join`
- Operation: `sum`
- Question: Calculate the total price of orders by Customer#000000013.
- Expected SQL:
```sql
SELECT SUM(T2.o_totalprice) FROM customer AS T1 INNER JOIN orders AS T2 ON T1.c_custkey = T2.o_custkey WHERE T1.c_name = 'Customer#000000013';
```
- Generated SQL:
```sql

```
- Schema valid: `False`
- Execution success: `False`
- Schema error: `empty_sql`
- Generated error: `No generated SQL candidate passed parse, selected-schema validation, and DuckDB EXPLAIN. This is a hard rejection, not an execution failure.`
