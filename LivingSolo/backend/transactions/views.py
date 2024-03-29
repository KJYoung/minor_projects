"""
    Transaction, Trxn, 거래에 관한 처리 로직입니다.
"""
import json
from json.decoder import JSONDecodeError
from calendar import monthrange

from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponseNotFound, JsonResponse
from transactions.models import Transaction
from tags.models import Tag
from tags.utils import get_tag_dict_from_obj_list
from transactions.utils import get_transaction_dict_from_obj


@require_http_methods(['GET', 'POST'])
def general_transaction(request):
    """
    GET : get element list
    POST : create element
    """
    if request.method == 'GET':
        # Params: keyword<String?>, year<Number?(Number if month is valid)>, month<Number?>

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

        result = [get_transaction_dict_from_obj(tr_elem) for tr_elem in searched_trxn]
        return JsonResponse({"elements": result}, safe=False)
    else:  ## post
        try:
            req_data = json.loads(request.body.decode())
            element = Transaction(
                memo=req_data["memo"],
                amount=-req_data["amount"] if req_data["isMinus"] else req_data["amount"],
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


@require_http_methods(['GET'])
def general_trxn_combined(request):
    """
    GET : get trxn list - Day Combined
    """
    # Params: year<Number>, month<Number>

    query_args = {}
    query_args["year"] = request.GET.get("year", None)
    query_args["month"] = request.GET.get("month", None)

    trxn_all = Transaction.objects.all()

    # Filtering
    # 1. Filter by Year & Month
    if query_args["year"] and query_args["month"]:
        filtered_trxn = trxn_all.filter(
            date__year=query_args["year"], date__month=query_args["month"]
        )
    else:
        return HttpResponseBadRequest()

    # result = [[]] *(monthrange(int(query_args["year"]), int(query_args["month"]))[1] + 1)
    # 위와 같이 하면 ex) result[27].append(~) 했을 때, 모든 Array에 append. 즉 각 element의 array reference가 같아진다.
    result = [
        {"amount": 0, "tag": []}
        for _ in range((monthrange(int(query_args["year"]), int(query_args["month"]))[1] + 1))
    ]
    for trxn_elem in filtered_trxn:
        result[trxn_elem.date.day]['amount'] += trxn_elem.amount

        for tag_elem in list(trxn_elem.tag.all().values()):
            tag_json = {
                "id": tag_elem['id'],
                "name": tag_elem['name'],
                "color": tag_elem['color'],
            }
            if tag_json in result[trxn_elem.date.day]['tag']:
                pass
            else:
                result[trxn_elem.date.day]['tag'].append(tag_json)
    return JsonResponse({"elements": result}, safe=False)


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
    else:  ## delete
        try:
            trxn_id = int(trxn_id)
            trxn_obj = Transaction.objects.get(pk=trxn_id)

            trxn_obj.delete()
            return JsonResponse({"message": "success"}, status=200)
        except Transaction.DoesNotExist:
            return HttpResponseNotFound()
