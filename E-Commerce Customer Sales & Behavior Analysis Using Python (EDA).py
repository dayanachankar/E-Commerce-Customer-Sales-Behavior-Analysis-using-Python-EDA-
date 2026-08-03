#!/usr/bin/env python
# coding: utf-8

# # Load Required Libraries

# In[322]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# # Load All 6 Datasets

# In[323]:


orders = pd.read_csv("C:/Users/HP/Downloads/olist_orders_dataset.csv")

order_items = pd.read_csv("C:/Users/HP/Downloads/olist_order_items_dataset.csv")

payments = pd.read_csv("C:/Users/HP/Downloads/olist_order_payments_dataset.csv")

customers = pd.read_csv("C:/Users/HP/Downloads/olist_customers_dataset.csv")

products = pd.read_csv("C:/Users/HP/Downloads/olist_products_dataset.csv")

category_translation = pd.read_csv("C:/Users/HP/Downloads/product_category_name_translation.csv")


# # Understand Each Dataset
# 

# In[324]:


# Shape

orders.shape


# In[325]:


order_items.shape


# In[326]:


payments.shape


# In[327]:


customers.shape


# In[328]:


products.shape


# In[329]:


category_translation.shape


# In[330]:


# View First 5 Rows
orders.head()


# In[331]:


order_items.head()


# In[332]:


payments.head()


# In[333]:


customers.head()


# In[334]:


products.head()


# In[335]:


category_translation.head()


# In[336]:


# Check Columns
orders.columns


# In[337]:


order_items.columns


# In[338]:


payments.columns


# In[339]:


customers.columns


# In[340]:


products.columns


# In[341]:


category_translation.columns


# # Data Quality Check

# In[342]:


# Check data type
orders.info()


# In[343]:


order_items.info()


# In[344]:


payments.info()


# In[345]:


customers.info()


# In[346]:


products.info()


# In[347]:


category_translation.info()


# In[348]:


#missing value

orders.isnull().sum()


# In[349]:


order_items.isnull().sum()


# In[350]:


payments.isnull().sum()


# In[351]:


customers.isnull().sum()


# In[352]:


products.isnull().sum()


# In[353]:


category_translation.isnull().sum()


# In[354]:


# Duplicate rows

orders.duplicated().sum()


# In[355]:


order_items.duplicated().sum()


# In[356]:


payments.duplicated().sum()


# In[357]:


customers.duplicated().sum()


# In[358]:


products.duplicated().sum()


# In[359]:


category_translation.duplicated().sum()


# In[360]:


# convert date column

##Convert Orders date columns

date_columns = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col])


# In[361]:


orders.info()


# In[362]:


## Convert Order Items date column

order_items['shipping_limit_date'] = pd.to_datetime(
    order_items['shipping_limit_date']
)


# In[363]:


order_items.info()


# ## Data cleaning

# In[364]:


## cleaned Product table dataset
# Handle Missing Product Category

products['product_category_name'].isnull().sum()


# In[365]:


products['product_category_name'] = products['product_category_name'].fillna('Unknown')


# In[366]:


products['product_category_name'].isnull().sum()


# In[367]:


# Handle Missing Numerical Columns
product_columns = [
    'product_name_lenght',
    'product_description_lenght',
    'product_photos_qty',
    'product_weight_g',
    'product_length_cm',
    'product_height_cm',
    'product_width_cm'
]

for col in product_columns:
    products[col] = products[col].fillna(products[col].median())


# In[368]:


products.isnull().sum()


# In[369]:


### cleaned Order table dataset

# check order status
orders['order_status'].value_counts()


# In[370]:


#Check Missing Delivery Dates by Status
orders[orders['order_delivered_customer_date'].isnull()]['order_status'].value_counts()


# In[371]:


#For missing carrier date:
orders[orders['order_delivered_carrier_date'].isnull()]['order_status'].value_counts()


# In[372]:


# Keep only delivered orders — cancelled/unavailable orders aren't real sales
orders = orders[orders['order_status'] == 'delivered'].copy()


# In[373]:


print(orders.shape)


# In[374]:


print(orders['order_status'].value_counts())


# # Data Integration (Merge Table)

# In[375]:


df = orders.merge(customers, on='customer_id', how='left')


# In[376]:


df = df.merge(order_items, on='order_id', how='left', validate="1:m")


# In[377]:


df = df.merge(products, on='product_id', how='left')


# In[378]:


df = df.merge(category_translation, on='product_category_name', how='left')


# In[379]:


# Aggregate payments to ONE row per order before merging.
# payments has multiple rows per order (installments) — merging it as-is would
# duplicate order_items rows and inflate revenue numbers.

payment_summary = payments.groupby('order_id').agg(
    payment_type=('payment_type', lambda x: x.mode().iloc[0]),
    total_payment_value=('payment_value', 'sum'),
).reset_index()
 


# In[380]:


df = df.merge(payment_summary, on='order_id', how='left', validate='m:1')


# In[381]:


df.shape


# In[382]:


df.head()


# In[383]:


df.info()


# ## Post-Merge Data Validation

# In[384]:


print(df.shape)


# In[385]:


df.info()


# In[386]:


df.isnull().sum()


# In[387]:


df.duplicated().sum()


# In[388]:


df.head()


# ## Feature Engineering

# In[389]:


#Revenue
df['revenue'] = df['price'] + df['freight_value']


# In[390]:


df[['price','freight_value','revenue']].head()


# In[391]:


## Delivery Days
df['delivery_days'] = (
    df['order_delivered_customer_date'] - 
    df['order_purchase_timestamp']
).dt.days


# In[392]:


##  Delivery Delay 
df['delivery_delay_days'] = (
    df['order_delivered_customer_date'] -
    df['order_estimated_delivery_date']
).dt.days


# In[393]:


## Purchase Year 
df['purchase_year'] = df['order_purchase_timestamp'].dt.year


# In[394]:


## Purchase Month 
df['purchase_month'] = df['order_purchase_timestamp'].dt.month


# In[395]:


## Purchase Day 
df['purchase_day'] = df['order_purchase_timestamp'].dt.day_name()


# In[396]:


# Check new columns created
df.columns


# In[397]:


# Check first few rows
df[['price',
    'freight_value',
    'revenue',
    'delivery_days',
    'delivery_delay_days',
    'purchase_year',
    'purchase_month',
    'purchase_day']].head()


# In[398]:


#Check missing values in new columns
df[['revenue',
    'delivery_days',
    'delivery_delay_days']].isnull().sum()


# In[399]:


#check statistics
df[['revenue',
    'delivery_days',
    'delivery_delay_days']].describe()


# ## Exploratory Data Analysis (EDA)

# # Analysis 1: Sales Performance Analysis

# In[400]:


# Total Revenue -What is the total revenue generated by the company?

total_revenue = df['revenue'].sum()

print("Total Revenue:", total_revenue)


# In[401]:


# Total Orders - How many orders did customers place?

total_orders = df['order_id'].nunique()

print("Total Orders:", total_orders)


# In[402]:


# Average Order Value - How much does a customer spend on average per order?

average_order_value = total_revenue / total_orders

print("Average Order Value:", average_order_value)


# In[403]:


# Monthly Revenue Trend - How is revenue changing month by month?

monthly_revenue = df.groupby(
    ['purchase_year','purchase_month']
)['revenue'].sum().reset_index()

monthly_revenue.head()


# In[404]:


# Create Month-Year Column
monthly_revenue['year_month'] = (
    monthly_revenue['purchase_year'].astype(str) 
    + '-' +
    monthly_revenue['purchase_month'].astype(str)
)

monthly_revenue.head()


# In[405]:


# Monthly Revenue Trend


plt.figure(figsize=(12,5))

sns.lineplot(
    data=monthly_revenue,
    x='year_month',
    y='revenue',
    marker='o'
)

plt.xticks(rotation=45)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.show()


# # Analysis 2: Product Performance Analysis

# In[406]:


#Top 10 Product- Which product categories contribute the highest revenue?

category_revenue = df.groupby(
    'product_category_name_english'
)['revenue'].sum().sort_values(
    ascending=False
).head(10).reset_index()

category_revenue


# In[407]:


plt.figure(figsize=(10,5))

sns.barplot(
    data=category_revenue,
    x='revenue',
    y='product_category_name_english'
)

plt.title("Top 10 Product Categories by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product Category")

plt.show()


# In[408]:


# top Selling Product Categories -Which categories have the highest number of items sold?

category_sales = df.groupby(
    'product_category_name_english'
)['product_id'].count().sort_values(
    ascending=False
).head(10).reset_index()

category_sales


# In[409]:


plt.figure(figsize=(10,5))

sns.barplot(
    data=category_sales,
    x='product_id',
    y='product_category_name_english'
)

plt.title("Top 10 Selling Product Categories")
plt.xlabel("Number of Products Sold")
plt.ylabel("Product Category")

plt.show()


# # Analysis 3 : Customer Behavior Analysis

# In[410]:


# Customer Distribution by State- Which states have the highest number of customers?

customer_state = df.groupby(
    'customer_state'
)['customer_unique_id'].nunique().sort_values(
    ascending=False
).head(10).reset_index()

customer_state


# In[411]:


plt.figure(figsize=(10,5))

sns.barplot(
    data=customer_state,
    x='customer_unique_id',
    y='customer_state'
)

plt.title("Top 10 States by Number of Customers")
plt.xlabel("Number of Customers")
plt.ylabel("State")

plt.show()


# In[412]:


# Revenue by State - Which states generate the highest revenue?

state_revenue = df.groupby(
    'customer_state'
)['revenue'].sum().sort_values(
    ascending=False
).head(10).reset_index()

state_revenue


# In[413]:


plt.figure(figsize=(10,5))

sns.barplot(
    data=state_revenue,
    x='revenue',
    y='customer_state'
)

plt.title("Top 10 States by Revenue")
plt.xlabel("Revenue")
plt.ylabel("State")

plt.show()


# # Analysis 4 : Payment Analysis

# In[414]:


# Payment Method Distribution-How do customers prefer to pay?
payment_distribution = df.groupby(
    'payment_type'
)['order_id'].count().reset_index()

payment_distribution


# In[415]:


plt.figure(figsize=(8,5))

sns.barplot(
    data=payment_distribution,
    x='payment_type',
    y='order_id'
)

plt.title("Customer Payment Method Preference")
plt.xlabel("Payment Type")
plt.ylabel("Number of Orders")

plt.show()


# In[416]:


# Revenue by Payment Type -Which payment method contributes the highest revenue?

payment_revenue = df.groupby(
    'payment_type'
)['total_payment_value'].sum().sort_values(
    ascending=False
).reset_index()

payment_revenue


# In[417]:


plt.figure(figsize=(8,5))

sns.barplot(
    data=payment_revenue,
    x='payment_type',
    y='total_payment_value'
)

plt.title("Revenue by Payment Method")
plt.xlabel("Payment Type")
plt.ylabel("Revenue")

plt.show()


# # Analysis 5: Delivery Performance Analysis

# In[418]:


# Delivery Time Distribution- How many days does delivery usually take?

delivery_time = df['delivery_days'].dropna()

delivery_time.describe()


# In[419]:


plt.figure(figsize=(10,5))

sns.histplot(
    data=delivery_time,
    bins=30
)

plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Days")
plt.ylabel("Number of Orders")

plt.show()


# In[420]:


# Early vs Late Delivery Analysis - Early vs Late Delivery Analysis

df['delivery_status'] = df['delivery_delay_days'].apply(
    lambda x: 'Late Delivery' if x > 0 else 'Early Delivery'
)


# In[421]:


df['delivery_status'].value_counts()


# In[422]:


delivery_status = df['delivery_status'].value_counts().reset_index()

delivery_status.columns = ['delivery_status','count']


plt.figure(figsize=(8,5))

sns.barplot(
    data=delivery_status,
    x='delivery_status',
    y='count'
)

plt.title("Early vs Late Delivery")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Orders")

plt.show()


# # Business Insights & Recommendations

# In[423]:


summary = {
    "Total Revenue": df['revenue'].sum(),
    "Total Orders": df['order_id'].nunique(),
    "Average Order Value": df['revenue'].sum()/df['order_id'].nunique(),
    "Average Delivery Days": df['delivery_days'].mean()
}

summary


# 
# ### Business Insights:
# 
# - Generated **15.42M** revenue from **96K+** customer orders.
# - Average Order Value (AOV) is **159.83** per order.
# - Top product categories contribute the highest share of revenue.
# - Customer demand is concentrated in a few states.
# - Credit Card is the most preferred payment method.
# - Average delivery time is approximately **12 days**.
# - Most orders are delivered on or before the estimated delivery date.

# In[ ]:




