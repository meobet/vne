import urllib
import pandas as pd

def main():
    data = pd.read_excel("shop/shop.xlsx")
    for i, row in data.iterrows():
        if i < 11633 or row.image_url.endswith("/"):
            continue
        print(i, row.title, row.image_url)
        urllib.request.urlretrieve(row.image_url, "shop/images/" + str(row.product_id) + "." + row.image_url.split(".")[-1])

if __name__ == "__main__":
    main()