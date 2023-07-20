from django.urls import path
from transactions.views import (
    general_transaction,
    detail_transaction,
)

urlpatterns = [
    path('', general_transaction, name="trxn"),
    # Params: combined<Boolean?>, keyword<String?>, year<Number?(Number if month is valid)>, month<Number?>
    path('<int:trxn_id>/', detail_transaction, name="trxn_detail"),
]
