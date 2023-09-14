"""
    Tag, 각종 모델에 사용되는 태그에 관한 처리 로직입니다.
"""

import json
from json.decoder import JSONDecodeError
from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, JsonResponse
from django.forms.models import model_to_dict
from tags.models import TagClass, Tag, TagPreset
from tags.utils import get_tag_dict_from_obj
from todos.utils import get_todo_category_and_color_dict_nullcheck

## TagClass
@require_http_methods(['GET', 'POST'])
def general_tag_class(request):
    """
    GET : get tag class list
    POST : create tag category(tag class)
    """
    if request.method == 'GET':
        try:
            result = []
            for trc_elem in TagClass.objects.all():
                tags_list = get_tag_dict_from_obj(list(trc_elem.tag.all().values()))
                result.append(
                    {
                        "id": trc_elem.id,
                        "name": trc_elem.name,
                        "color": trc_elem.color,
                        "tags": tags_list,
                    }
                )
            return JsonResponse(
                {
                    "elements": result,
                },
                safe=False,
            )
        except (TagClass.DoesNotExist):
            print("ERROR from general_tag_class")
            return HttpResponseBadRequest()
    else:  # POST REQUEST
        try:
            req_data = json.loads(request.body.decode())
            element = TagClass(
                name=req_data["name"],
                color=req_data["color"],
            )
            element.save()
        except (KeyError, JSONDecodeError):
            return HttpResponseBadRequest()
        return JsonResponse({"id": element.id, "name": element.name}, status=201)


## Tag
@require_http_methods(['GET', 'POST'])
def general_tag(request):
    """
    GET : get tag list
    POST : create tag category(tag class)
    """
    if request.method == 'GET':
        try:
            result = []
            for tr_elem in Tag.objects.all():
                class_elem = tr_elem.tag_class
                # tr_elem.tag_class(FK field) : TagClass Object
                result.append(
                    {
                        "id": tr_elem.id,
                        "name": tr_elem.name,
                        "color": tr_elem.color,
                        "tag_class": {
                            "id": class_elem.id,
                            "name": class_elem.name,
                            "color": class_elem.color,
                        },
                    }
                )
            return JsonResponse({"elements": result}, safe=False)
        except (KeyError, JSONDecodeError):
            print("ERROR from general_tag")
            return HttpResponseBadRequest()
    else:  # POST REQUEST
        try:
            req_data = json.loads(request.body.decode())
            tag_class = TagClass.objects.get(pk=req_data["class"])
            element = Tag(name=req_data["name"], color=req_data["color"], tag_class=tag_class)
            element.save()
        except (KeyError, JSONDecodeError):
            return HttpResponseBadRequest()
        return JsonResponse({"id": element.id, "name": element.name}, status=201)


## Tag Detail
@require_http_methods(['GET'])
def tag_detail(request, tag_id):
    """
    GET : get tag detail
    """
    if request.method == 'GET':
        try:
            tag_id = int(tag_id)
            tag_obj = Tag.objects.get(pk=tag_id)

            trxns, todos = [], []
            for trxn_elem in tag_obj.transaction.all():
                tags = get_tag_dict_from_obj(list(trxn_elem.tag.values()))

                trxn_elem = model_to_dict(trxn_elem)
                trxns.append(
                    {
                        "id": trxn_elem["id"],
                        "memo": trxn_elem["memo"],
                        "date": trxn_elem["date"],
                        "tag": tags,
                        "period": trxn_elem["period"],
                        "amount": trxn_elem["amount"],
                    }
                )

            for todo_elem in tag_obj.todo.all():
                tags = get_tag_dict_from_obj(list(todo_elem.tag.values()))
                categ_json, categ_color = get_todo_category_and_color_dict_nullcheck(
                    todo_elem.category
                )

                todos.append(
                    {
                        "id": todo_elem.id,
                        "name": todo_elem.name,
                        "tag": tags,
                        "done": todo_elem.done,
                        "color": categ_color,
                        "category": categ_json,
                        "priority": todo_elem.priority,
                        "deadline": todo_elem.deadline,
                        "is_hard_deadline": todo_elem.is_hard_deadline,
                        "period": todo_elem.period,
                    }
                )

            result = {
                "transaction": trxns,
                "todo": todos,
            }
            return JsonResponse({"elements": result}, safe=False)
        except (KeyError, JSONDecodeError):
            print("ERROR from tag_detail")
            return HttpResponseBadRequest()
    else:
        return HttpResponseBadRequest()


## Tag Preset
@require_http_methods(['GET', 'POST'])
def general_tag_preset(request):
    """
    GET : get tag preset list
    POST : create tag preset
    """
    if request.method == 'GET':
        try:
            result = []
            for tp_elem in TagPreset.objects.all():
                tags_list = get_tag_dict_from_obj(list(tp_elem.tags.all().values()))
                result.append(
                    {
                        "id": tp_elem.id,
                        "name": tp_elem.name,
                        "tags": tags_list,
                    }
                )
            return JsonResponse({"elements": result}, safe=False)
        except (KeyError, JSONDecodeError):
            print("ERROR from general_tagpreset")
            return HttpResponseBadRequest()
    else:  # POST REQUEST
        try:
            req_data = json.loads(request.body.decode())
            element = TagPreset(name=req_data["name"])
            element.save()

            # Set tags
            tag_list = req_data["tags"]
            for tag in tag_list:
                tag_elem = Tag.objects.get(pk=tag["id"])
                element.tags.add(tag_elem)
        except (KeyError, JSONDecodeError, Tag.DoesNotExist):
            return HttpResponseBadRequest()
        return JsonResponse({"id": element.id, "name": element.name}, status=201)
