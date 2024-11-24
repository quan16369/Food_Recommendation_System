import requests

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/CBCVVX4wJjXG64DKYMVi1w/FoodDataSet.js"
response = requests.get(url)

with open("FoodDataSet.js", "wb") as file:
    file.write(response.content)

print("Download completed!")
