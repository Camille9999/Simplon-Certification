from django import forms

class ShapefileForm(forms.Form):
    vcub_config = forms.CharField()
    ep_config = forms.CharField()
