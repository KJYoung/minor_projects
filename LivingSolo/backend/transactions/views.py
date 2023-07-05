import json

from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponseNotFound, JsonResponse
from transactions.models import Transaction, TransactionType

from json.decoder import JSONDecodeError


@require_http_methods(['GET', 'POST'])
def general_transaction(request):
    """
    GET : get element list
    POST : create element
    """
    if request.method == 'GET':
        result = []
        for tr_elem in Transaction.objects.all():
            types = []
            for type_elem in list(tr_elem.type.all().values()):
                types.append(
                    {
                        "id": type_elem['id'],
                        "name": type_elem['name'],
                        "color": type_elem['color'],
                    }
                )
            result.append(
                {
                    "id": tr_elem.id,
                    "memo": tr_elem.memo,
                    "date": tr_elem.date,
                    "type": types,
                    "period": tr_elem.period,
                    "amount": tr_elem.amount,
                }
            )
        return JsonResponse({"elements": result}, safe=False)
    else:  ## post
        try:
            req_data = json.loads(request.body.decode())

            element = Transaction(
                memo=req_data["memo"],
                amount=req_data["amount"],
                period=req_data["period"],
                date=req_data["date"],
            )
            element.save()
        except (KeyError, JSONDecodeError):
            return HttpResponseBadRequest()
        return JsonResponse({"id": element.id, "memo": element.memo}, status=201)


@require_http_methods(['PUT', 'DELETE'])
def detail_transaction(request, element_id):
    """
    PUT : edit element's content
    DELETE : delete element
    """
    if request.method == 'PUT':
        try:
            data = json.loads(request.body.decode())
            core_id = int(element_id)
            core_obj = Transaction.objects.get(pk=core_id)

            core_obj.name = data["name"]
            core_obj.save()
            return JsonResponse({"message": "success"}, status=200)
        except Transaction.DoesNotExist:
            return HttpResponseNotFound()
        except Exception:
            return HttpResponseBadRequest()
    else:  ## delete
        try:
            core_id = int(element_id)
            core_obj = Transaction.objects.get(pk=core_id)

            core_obj.delete()
            return JsonResponse({"message": "success"}, status=200)
        except Transaction.DoesNotExist:
            return HttpResponseNotFound()
        except Exception:
            return HttpResponseBadRequest()


## Transaction Type
@require_http_methods(['GET', 'POST'])
def general_trxn_type(request):
    """
    GET : get trxn type list
    POST : create new type
    """
    if request.method == 'GET':
        result = []
        for tr_elem in TransactionType.objects.all():
            result.append(
                {
                    "id": tr_elem.id,
                    "name": tr_elem.name,
                }
            )
        return JsonResponse({"elements": result}, safe=False)
    else:  ## post
        try:
            req_data = json.loads(request.body.decode())

            element = TransactionType(
                name=req_data["name"],
            )
            element.save()
        except (KeyError, JSONDecodeError):
            return HttpResponseBadRequest()
        return JsonResponse({"id": element.id, "name": element.name}, status=201)
