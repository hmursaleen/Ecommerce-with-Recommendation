from django import forms
from .models import Product, Order, Comment
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
 



class OrderForm(forms.ModelForm):
    class Meta:
        model = Order
        fields = ['first_name', 'last_name', 'address', 'zipcode', 'city']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        for field_name, field in self.fields.items():
            field.widget.attrs.update({
                'class': 'form-control',
                'style': 'width: 100%; padding: 10px; box-sizing: border-box;'
            })

        self.fields['first_name'].widget.attrs.update({'placeholder': 'First Name'})
        self.fields['last_name'].widget.attrs.update({'placeholder': 'Last Name'})
        self.fields['address'].widget.attrs.update({'placeholder': 'Address'})
        self.fields['zipcode'].widget.attrs.update({'placeholder': 'Zip Code'})
        self.fields['city'].widget.attrs.update({'placeholder': 'City'})







class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['category', 'title', 'description', 'price', 'image', 'thumbnail']

        widgets = {
		'category' : forms.Select(attrs={
			'class' : 'w-full p-4 border border-gray-200 rounded-xl'
			}),

		'title' : forms.TextInput(attrs={
			'class' : 'w-full p-4 border border-gray-200 rounded-xl'
			}),

		'description' : forms.Textarea(attrs={
			'class' : 'w-full p-4 border border-gray-200 rounded-xl'
			}),

		'price' : forms.TextInput(attrs={
			'class' : 'w-full p-4 border border-gray-200 rounded-xl'
			}),

		'image' : forms.FileInput(attrs={
			'class' : 'w-full p-4 border border-gray-200 rounded-xl'
			}),
		}





class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['text']
        widgets = {
            'text': forms.Textarea(attrs={
                'placeholder': 'Share your thoughts...',
                'class': 'w-full px-4 py-2 text-gray-700 bg-white border rounded-lg focus:outline-none focus:ring focus:border-blue-300',
                'rows': 3,
            }),
        }







class RatingForm(forms.Form):
    rating = forms.DecimalField(label='Rating', max_digits=3, decimal_places=1, min_value=0, max_value=10)