from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=100)
    sku = models.CharField(max_length=50, unique=True)
    category = models.CharField(max_length=100)
    stock_quantity = models.IntegerField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.sku})"

class SalesRecord(models.Model):
    date = models.DateField()
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity_sold = models.IntegerField()
    revenue = models.FloatField()
    is_holiday = models.BooleanField(default=False)
    temperature = models.FloatField(null=True, blank=True)
    unemployment = models.FloatField(null=True, blank=True)
    fuel_price = models.FloatField(null=True, blank=True)

