"""
    Transaction, Trxn, 거래에 관한 처리 로직입니다.
"""
import json
from json.decoder import JSONDecodeError

from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponseNotFound, JsonResponse
from transactions.models import Transaction
from tags.models import Tag


@require_http_methods(['GET', 'POST'])
def general_transaction(request):
    """
    GET : get element list
    POST : create element
    """
    if request.method == 'GET':
        # Params: combined<Boolean?>, keyword<String?>, year<Number?(Number if month is valid)>, month<Number?>

        query_args = {}
        query_args["combined"] = bool(request.GET.get("combined", False))
        query_args["keyword"] = request.GET.get("keyword", None)
        query_args["year"] = request.GET.get("year", None)
        query_args["month"] = request.GET.get("month", None)

        searched_trxn = Transaction.objects.all()
        # Filtering
        # 1. Filter by Year & Month
        if query_args["year"] and query_args["month"]:
            # 1-1. Both Year & Month
            searched_trxn = searched_trxn.filter(
                date__year=query_args["year"], date__month=query_args["month"]
            )
        elif query_args["year"]:
            # 1-2. Only Year
            searched_trxn = searched_trxn.filter(date__year=query_args["year"])

        # 2. Filter by Keyword
        if query_args["keyword"]:
            searched_trxn = searched_trxn.filter(memo__icontains=query_args["keyword"])

        result = []
        for tr_elem in searched_trxn:
            tags = []
            for tag_elem in list(tr_elem.tag.all().values()):
                tags.append(
                    {
                        "id": tag_elem['id'],
                        "name": tag_elem['name'],
                        "color": tag_elem['color'],
                    }
                )
            result.append(
                {
                    "id": tr_elem.id,
                    "memo": tr_elem.memo,
                    "date": tr_elem.date,
                    "tag": tags,
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

            # Set tags
            tag_bubble_list = req_data["tag"]
            for tag_bubble in tag_bubble_list:
                tag_elem = Tag.objects.get(pk=tag_bubble["id"])
                element.tag.add(tag_elem)
        except (KeyError, JSONDecodeError):
            return HttpResponseBadRequest()
        return JsonResponse({"id": element.id, "memo": element.memo}, status=201)


@require_http_methods(['PUT', 'DELETE'])
def detail_transaction(request, trxn_id):
    """
    PUT : edit element's content
    DELETE : delete element
    """
    if request.method == 'PUT':
        try:
            data = json.loads(request.body.decode())
            trxn_id = int(trxn_id)
            trxn_obj = Transaction.objects.get(pk=trxn_id)

            trxn_obj.memo = data["memo"]
            trxn_obj.amount = data["amount"]
            trxn_obj.save()
            return JsonResponse({"message": "success"}, status=200)
        except Transaction.DoesNotExist:
            return HttpResponseNotFound()
        except Exception:
            return HttpResponseBadRequest()
    else:  ## delete
        try:
            trxn_id = int(trxn_id)
            trxn_obj = Transaction.objects.get(pk=trxn_id)

            trxn_obj.delete()
            return JsonResponse({"message": "success"}, status=200)
        except Transaction.DoesNotExist:
            return HttpResponseNotFound()
        except Exception:
            return HttpResponseBadRequest()
