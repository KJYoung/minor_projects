import csv

from transactions.models import trxn_fields
from django.http import HttpResponse
from transactions.models import Transaction
from tags.utils import get_tag_dict_from_obj_list


def export_trxn_csv(request):
    response = HttpResponse(content_type='text/csv; charset=EUC-KR')
    response['Content-Disposition'] = 'attachment; filename="trxn.csv"'

    writer = csv.writer(response)
    writer.writerow(trxn_fields)

    trxns = Transaction.objects.all().values_list(*trxn_fields)
    for trxn in trxns:
        writer.writerow(trxn)

    return response


# Return Transaction Dictionary from Django Transaction Object
def get_transaction_dict_from_obj(transaction_obj):
    tags = get_tag_dict_from_obj_list(list(transaction_obj.tag.all().values()))
    return {
        "id": transaction_obj.id,
        "memo": transaction_obj.memo,
        "date": transaction_obj.date,
        "tag": tags,
        "period": transaction_obj.period,
        "amount": transaction_obj.amount,
    }
