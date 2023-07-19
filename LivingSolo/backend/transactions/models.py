from django.db import models
from core.models import AbstractTimeStampedModel


class TransactionTypeClass(AbstractTimeStampedModel):
    """Transaction Type Class definition"""

    name = models.CharField(max_length=30, null=False)
    color = models.CharField(max_length=7, null=False)

    def __str__(self):
        """To string method"""
        return str(self.name)


class TransactionType(AbstractTimeStampedModel):
    """Transaction Type definition"""

    name = models.CharField(max_length=30, null=False)
    color = models.CharField(max_length=7, null=False)
    type_class = models.ForeignKey(
        'TransactionTypeClass', related_name='type', on_delete=models.CASCADE, null=True
    )

    def __str__(self):
        """To string method"""
        return str(self.name)


# Create your models here.
class Transaction(AbstractTimeStampedModel):
    """Transaction definition"""

    date = models.DateTimeField(blank=True)
    type = models.ManyToManyField(TransactionType, related_name="transaction")
    amount = models.IntegerField()
    memo = models.CharField(max_length=200, null=False)

    period = models.IntegerField(default=0)  # 0 means Not Periodic Transaction.

    class Meta:
        ordering = ("-date",)
