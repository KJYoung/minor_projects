"""
    Tag, 각종 모델에 사용되는 태그에 관한 처리 로직입니다.
"""

import json
from json.decoder import JSONDecodeError
from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, JsonResponse, HttpResponseNotFound
from django.forms.models import model_to_dict
from tags.models import TagClass, Tag, TagPreset
from tags.utils import (
    get_tag_dict_from_obj_list,
    get_tag_preset_dict_from_obj,
    get_tag_class_dict_from_obj,
    get_tag_dict_from_tag_obj,
)
from todos.utils import get_todo_dict_from_tag_obj
from transactions.utils import get_transaction_dict_from_obj

## TagClass
@require_http_methods(['GET', 'POST'])
def general_tag_class(request):
    """
    GET : get tag class list
    POST : create tag category(tag class)
    """
    if request.method == 'GET':
        try:
            result = [get_tag_class_dict_from_obj(tc_elem) for tc_elem in TagClass.objects.all()]
            return JsonResponse({"elements": result}, safe=False)
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
            result = [get_tag_dict_from_tag_obj(t_elem) for t_elem in Tag.objects.all()]
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

            trxns = [
                get_transaction_dict_from_obj(trxn_elem) for trxn_elem in tag_obj.transaction.all()
            ]
            todos = [get_todo_dict_from_tag_obj(todo_elem) for todo_elem in tag_obj.todo.all()]

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
            result = [get_tag_preset_dict_from_obj(tp_elem) for tp_elem in TagPreset.objects.all()]
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


@require_http_methods(['PUT', 'DELETE'])
def tag_preset_detail(request, tag_preset_id):
    """
    PUT : edit tag preset
    DELETE : delete tag preset
    """
    tag_preset_id = int(tag_preset_id)

    try:
        tag_preset_obj = TagPreset.objects.get(pk=tag_preset_id)

        if request.method == 'PUT':
            return NotImplementedError()
        else:  # DELETE REQUEST
            tag_preset_obj.delete()
            return JsonResponse({"message": "success"}, status=200)
    except TagPreset.DoesNotExist:
        return HttpResponseNotFound()
