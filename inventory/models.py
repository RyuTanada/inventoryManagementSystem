from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=255)
    sku = models.CharField(max_length=100, unique=True)
    category = models.CharField(max_length=100)
    stock_quantity = models.IntegerField(default=0)
    price = models.FloatField()
    last_updated = models.DateTimeField(auto_now=True)

class SalesRecord(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    date = models.DateField()
    quantity_sold = models.IntegerField()
    revenue = models.FloatField()
    is_holiday = models.BooleanField(default=False)
    temperature = models.FloatField(null=True, blank=True)
    unemployment = models.FloatField(null=True, blank=True)
    fuel_price = models.FloatField(null=True, blank=True)
    cpi = models.FloatField(null=True, blank=True)
