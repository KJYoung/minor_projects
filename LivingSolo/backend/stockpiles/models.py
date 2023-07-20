from django.db import models
from core.models import AbstractTimeStampedModel
from tags.models import Tag


class Stockpile(AbstractTimeStampedModel):
    """Stockpile definition"""

    name = models.CharField(max_length=40, null=False)
    image = models.CharField(max_length=255, null=False)  # URI for Image.
    amount = models.IntegerField()
    type = models.ManyToManyField(Tag, related_name="stockpile")
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
