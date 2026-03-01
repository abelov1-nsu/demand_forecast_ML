import pandas as pd
import numpy as np
from faker import Faker
from datetime import timedelta, datetime

np.random.seed(42)

def generate_products(num_products=10):
    fake = Faker("ru_RU")

    products = []
    for i in range(1, num_products+1):
        adjective = fake.word()
        noun = fake.word()      

        name = f"{adjective.capitalize()} {noun.capitalize()}"
        size = np.random.uniform(0.5, 2.0)  # условная размерность
        weight = np.random.uniform(100, 2000)  # граммы
        items_in_pack = np.random.randint(1, 12)
        price = np.round(np.random.uniform(5, 50), 2)
        discounted_price = np.round(price * np.random.uniform(0.8, 1.0), 2)
        promotion = np.random.choice([0,1], p=[0.7, 0.3])
        products.append({
            'id': f"P{i:03d}",
            'name': f"{name}",
            'price': price,
            'discounted_price': discounted_price,
            'promotion': promotion,
            'size': size,
            'weight': weight,
            'items_in_pack': items_in_pack
        })
    return pd.DataFrame(products)

def generate_sales_for_product(product, days=360, max_outliers=5):
    base_demand = 50
    dates = pd.date_range(datetime.today() - timedelta(days=days), periods=days)

    # случайная сезонность для товара
    season_amp = np.random.uniform(0.2, 0.5)
    season_phase = np.random.uniform(0, 2*np.pi)
    seasonality = 1 + season_amp * np.sin(2*np.pi*np.arange(days)/30 + season_phase)

    # коррекция по характеристикам товара
    size_factor = 1 / product['size']
    weight_factor = 1 / (product['weight']/1000)
    pack_factor = 1 / product['items_in_pack']
    promo_factor = 1 + 0.2 * product['promotion']

    # основной тренд с шумом
    sales = base_demand * seasonality * size_factor * weight_factor * pack_factor * promo_factor
    sales = np.round(sales + np.random.normal(0, 3, days)).astype(int)

    df = pd.DataFrame({
        'id': product['id'],
        'name': product['name'],
        'price': product['price'],
        'discounted_price': product['discounted_price'],
        'promotion': product['promotion'],
        'size': product['size'],
        'weight': product['weight'],
        'items_in_pack': product['items_in_pack'],
        'date': dates,
        'sales': sales
    })

    # добавляем выбросы
    num_outliers = np.random.randint(0, max_outliers+1)
    if num_outliers > 0:
        outlier_days = np.random.choice(days, num_outliers, replace=False)
        multipliers = np.random.uniform(1.5, 10, num_outliers)  # не ниже 1.5 чтобы был реальный пик
        df.loc[outlier_days, 'sales'] = np.round(df.loc[outlier_days, 'sales'] * multipliers).astype(int)

    return df

def generate_sales_data(num_products=10, days=360):
    products = generate_products(num_products)
    all_sales = []
    for _, product in products.iterrows():
        df_sales = generate_sales_for_product(product, days)
        all_sales.append(df_sales)
    return pd.concat(all_sales, ignore_index=True)

if __name__ == "__main__":
    df = generate_sales_data(num_products=10, days=720)
    df.to_csv("data/sales.csv", index=False)
    print("Синтетические данные сгенерированы: data/sales.csv")