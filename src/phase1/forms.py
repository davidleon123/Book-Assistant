from django import forms


class QuestionForm(forms.Form):
    question = forms.CharField(
        label="",
        max_length=500,
        widget=forms.Textarea(attrs={"rows": 10, "cols": 50})
    )
    