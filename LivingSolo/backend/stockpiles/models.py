from django.db import models
from core.models import AbstractTimeStampedModel


class StockpileTypeClass(AbstractTimeStampedModel):
    """StockpileTypeClass Type definition"""

    name = models.CharField(max_length=30, null=False)
    color = models.CharField(max_length=7, null=False)

    def __str__(self):
        """To string method"""
        return str(self.name)


class StockpileType(AbstractTimeStampedModel):
    """StockpileType Type definition"""

    name = models.CharField(max_length=30, null=False)
    color = models.CharField(max_length=7, null=False)
    type_class = models.ForeignKey(
        StockpileTypeClass, related_name='type', on_delete=models.CASCADE, null=True
    )

    def __str__(self):
        """To string method"""
        return str(self.name)


class Stockpile(AbstractTimeStampedModel):
    """Stockpile definition"""

    name = models.CharField(max_length=40, null=False)
    image = models.CharField(max_length=255, null=False)  # URI for Image.
    amount = models.IntegerField()
    type = models.ManyToManyField(StockpileType, related_name="stockpile")
    memo = models.CharField(max_length=100, null=False)

    # Last_purchased Day, Estimated "Out-of-stock" Day

    class Meta:
        ordering = ("name",)


class StockpileTransaction(AbstractTimeStampedModel):
    """Stockpile definition"""

    target = models.ForeignKey(
        Stockpile, related_name='stockpile_transaction', on_delete=models.CASCADE, null=False
    )
    delta_amount = models.IntegerField()
    memo = models.CharField(max_length=50, null=False)
