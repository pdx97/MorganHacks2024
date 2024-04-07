from django.shortcuts import render
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet
from django.conf import settings

from .models import *
from rest_framework.response import Response


class UserFormView(APIView):
    def post(self, request):
        data = request.data
        fields = data["data"]["fields"]
        formatedData = {}
        for field in fields:
            print("{")
            for key in field.keys():
                if key == "label" and field["label"] == "First Name":
                    formatedData["name"] = field["value"]
                if key == "label" and field["label"] == "Last Name":
                    formatedData["name"] = formatedData["name"] + field["value"]
                if key == "type" and field["type"] == "INPUT_EMAIL":
                    formatedData["email"] = field["value"]
                if key == "type" and field[key] == "INPUT_PHONE_NUMBER":
                    formatedData["phone"] = field["value"]
                if key == "key" and field[key] == "question_6DOYVY":
                    formatedData["currentJob"] = []
                    for option in field["options"]:
                        if option["id"] in field["value"]:
                            formatedData["currentJob"].append(option["text"])
                if key == "key" and field[key] == "question_7XVYP0":
                    formatedData["skills"] = []
                    for option in field["options"]:
                        if option["id"] in field["value"]:
                            formatedData["skills"].append(option["text"])

                if key == "key" and field[key] == "question_GeJavZ":
                    formatedData["experience"] = field["value"]
                if key == "key" and field[key] == "question_VpMXEg":
                    formatedData["education"] = []
                    for option in field["options"]:
                        if option["id"] in field["value"]:
                            formatedData["education"].append(option["text"])

                print(formatedData)

                if key != "options":
                    print(key + ": ", end="")
                    print(field[key])
                else:
                    print(key + ": " , end= "")
                    print(field[key][0])
            print("}")


        return Response("Success")



class TimelineView(APIView):
    def get(self, request):
        output = {
            'jobs':[{
                'job': "Software Eng",
                "skills":["python", "html", "react", "javascript"]
            },
                {
                    'job': "Flouriest",
                    'skills':['cutting', 'watering', "growing"]
                }]
        }
        return Response(output)

