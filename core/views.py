from django.conf import settings
from django.contrib import messages
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.utils.text import slugify
from django.db.models import Q, Count
from .models import Product, Category, UserItemInteraction, Order, OrderItem
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
from django.http import JsonResponse
from .cart import Cart
from .forms import OrderForm


import stripe
stripe.api_key = "sk_test_51NUXlWSDhw8NOhRD4Tck7ihluuTThDgMczuY7trorZ6k3iFYVjsutzMErqeOc9M06Geueni0wTvIMT9hN5rdnXo200aHeDSRRQ"



def home(request):
    products = Product.objects.all()
    return render(request, 'home.html', {
        'products' : products,
        })





def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Log the user in immediately after signup
            login(request, user)
            return redirect('home')  # Replace 'home' with the name of your home page URL pattern
    else:
        form = UserCreationForm()
    
    return render(request, 'signup.html', {'form': form})








def add_to_cart(request, product_id):
    cart = Cart(request)
    cart.add(product_id)

    return redirect('cart_view')




def success(request):
    return render(request, 'success.html')




def change_quantity(request, product_id):
    action = request.GET.get('action', '')

    if action:
        quantity = 1
        if action == 'decrease':
            quantity = -1

        cart = Cart(request)
        cart.add(product_id, quantity, True)

    return redirect('cart_view')




def remove_from_cart(request, product_id):
    cart = Cart(request)
    cart.remove(product_id)

    return redirect('cart_view')




def cart_view(request):
    cart = Cart(request)

    return render(request, 'cart_view.html', {
        'cart': cart
    })





@login_required
def checkout(request):
    cart = Cart(request)

    if cart.get_total_cost() == 0:
        return redirect('cart_view')

    if request.method == 'POST':
        data = json.loads(request.body)
        form = OrderForm(request.POST)

        total_price = 0
        items = []

        for item in cart:
            product = item['product']
            total_price += product.price * int(item['quantity'])

            items.append({
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': product.title,
                    },
                    'unit_amount': product.price
                },
                'quantity': item['quantity']
            })
        
        stripe.api_key = settings.STRIPE_SECRET_KEY
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=items,
            mode='payment',
            success_url=f'{settings.WEBSITE_URL}cart/success/',
            cancel_url=f'{settings.WEBSITE_URL}cart/',
        )
        payment_intent = session.payment_intent

        order = Order.objects.create(
            first_name=data['first_name'],
            last_name=data['last_name'],
            address=data['address'],
            zipcode=data['zipcode'],
            city=data['city'],
            created_by = request.user,
            is_paid = True,
            payment_intent = payment_intent,
            paid_amount = total_price
        )

        for item in cart:
            product = item['product']
            quantity = int(item['quantity'])
            price = product.price * quantity

            item = OrderItem.objects.create(order=order, product=product, price=price, quantity=quantity)

        cart.clear()

        return JsonResponse({'session': session, 'order': payment_intent})
    else:
        form = OrderForm()

    return render(request, 'checkout.html', {
        'cart': cart,
        'form': form,
        'pub_key': settings.STRIPE_PUB_KEY,
    })






def search(request):
    query = request.GET.get('query', '')
    products = Product.objects.filter(Q(title__icontains=query) | Q(slug__icontains=query))
    return render(request, 'search.html', {
        'query' : query,
        'products' : products,
        })





import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


import re

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stop Word Removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Punctuation Removal and Special Character Removal
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    tokens = [token for token in tokens if token]  # Remove empty tokens
    
    # Numerical Value Conversion
    tokens = ['NUM' if token.isdigit() else token for token in tokens]
    
    # Stemming and Lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens]
    
    # Spell Checking and Correction (requires external libraries)
    # Handle HTML Tags and URLs (requires additional code)
    # Removing Emojis and Special Symbols (requires additional code)
    
    # Whitespace Removal
    tokens = [token.strip() for token in tokens]
    
    # Handling Contractions (requires additional code)
    # Normalization (requires additional code)
    
    # Removing Short and Long Words
    tokens = [token for token in tokens if len(token) > 1 and len(token) < 15]
    
    # Removing Duplicates
    tokens = list(set(tokens))
    
    # Token Filtering (requires additional code)
    
    # N-gram Generation (requires additional code)
    
    # Handling Abbreviations and Acronyms (requires additional code)
    
    # Language Translation and Transliteration (requires external libraries)
    
    return tokens



def item_based_collaborative_filtering(product):
    all_interactions = UserItemInteraction.objects.filter(product__category=product.category)
    all_interactions = all_interactions.exclude(product=product)
    # Create a dictionary to store product descriptions for each product
    product_descriptions = {}
    for interaction in all_interactions:
        product_id = interaction.product.id
        product_description = interaction.product.description
        if product_id not in product_descriptions:
            product_descriptions[product_id] = product_description

    product_descriptions[product.id] = product.description
    

    tfidf_vectorizer = TfidfVectorizer()


    product_ids = []
    product_texts = []
    for product_id, descriptions in product_descriptions.items():
        product_ids.append(product_id)
        description = preprocess_text(descriptions)
        product_texts.append(" ".join(description))

    tfidf_matrix = tfidf_vectorizer.fit_transform(product_texts)

    # Calculate the cosine similarity between the target product and all other products
    target_product_index = product_ids.index(product.id)
    cosine_similarities = cosine_similarity(tfidf_matrix[target_product_index], tfidf_matrix)

    # Get similarity scores for the target product
    similarity_scores = cosine_similarities[0]

    # Sort product IDs by similarity scores in descending order
    sorted_product_ids = [x for _, x in sorted(zip(similarity_scores, product_ids), reverse=True)]

    # Get the top N most similar product IDs (excluding the target product)
    top_similar_product_ids = sorted_product_ids[:5]

    # Get the actual Product objects for the recommended product IDs
    recommended_products = Product.objects.filter(id__in=top_similar_product_ids)
    recommended_products = recommended_products.exclude(slug=product.slug)
    return recommended_products








def content_based_recommendation(user, product):
    viewed_interactions = UserItemInteraction.objects.filter(user=user)
    viewed_product_ids = list(viewed_interactions.values_list('product__id', flat=True))
    
    viewed_products = [interaction.product for interaction in viewed_interactions]
    
    # Combine product categories and prices into a single string
    user_profile = ' '.join(f'{product.category} {product.price}' for product in viewed_products)

    all_products = Product.objects.all()
    product_profiles = [' '.join([str(product.category), str(product.price)]) for product in all_products]
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([user_profile] + product_profiles)
    
    # Calculate the cosine similarity between the user profile and all product profiles
    cosine_similarities = linear_kernel(tfidf_matrix[0], tfidf_matrix[1:])
    
    # Create a list of tuples with product ID and similarity score
    product_similarity_pairs = [(all_products[i].id, cosine_similarities[0][i]) for i in range(len(all_products)) if all_products[i].id not in viewed_product_ids]
    
    # Sort the list of product-similarity pairs by similarity score in descending order
    sorted_product_similarity_pairs = sorted(product_similarity_pairs, key=lambda x: x[1], reverse=True)
    
    # Get recommended product IDs
    recommended_product_ids = [product_id for product_id, _ in sorted_product_similarity_pairs]
    
    # Retrieve recommended products
    recommended_products = [get_object_or_404(Product, id=x) for x in recommended_product_ids if x != product.id][:6]
    return recommended_products









def user_based_collaborative_filtering(target_user):
    # Get all interactions for all users
    all_interactions = UserItemInteraction.objects.all()

    # Create a dictionary to store product views for each user
    user_product_views = {}
    for interaction in all_interactions:
        user_id = interaction.user.id
        product_id = interaction.product.id
        if user_id not in user_product_views:
            user_product_views[user_id] = set()
        user_product_views[user_id].add(product_id)

    # Create a matrix where each row represents a user and each column represents a product
    num_users = User.objects.count()
    num_users += 10
    num_products = Product.objects.count()
    num_products += 10
    user_product_matrix = np.zeros((num_users, num_products))

    for user_id, product_ids in user_product_views.items():
        for product_id in product_ids:
            user_product_matrix[user_id - 1][product_id - 1] = 1  # Adjust indices

    # Calculate user similarity using cosine similarity
    similarity_matrix = cosine_similarity(user_product_matrix)

    # Get the user index for the target user
    target_user_index = target_user.id - 1

    # Get similarity scores for the target user
    user_similarities = similarity_matrix[target_user_index]

    # Normalize similarity scores between 0 and 1
    scaler = MinMaxScaler()
    user_similarities = scaler.fit_transform(user_similarities.reshape(-1, 1)).flatten()

    # Sort users based on similarity scores (in descending order)
    similar_users = sorted(range(len(user_similarities)), key=lambda i: user_similarities[i], reverse=True)

    # Get product recommendations for the target user from similar users' interactions
    recommended_products = set()
    for similar_user_index in similar_users:
        similar_user_interactions = user_product_views.get(similar_user_index + 1, set())
        recommended_products.update(similar_user_interactions)

    # Exclude products already viewed by the target user
    target_user_interactions = user_product_views.get(target_user.id, set())
    recommended_products -= target_user_interactions

    # Get the actual Product objects for the recommended product IDs
    recommended_products = Product.objects.filter(id__in=recommended_products)

    return recommended_products








def hybrid_recommendation(user, product):
    item_based_collaborative_filtering_results = item_based_collaborative_filtering(product)
    user_based_collaborative_filtering_results = user_based_collaborative_filtering(user)
    content_based_recommendation_results = content_based_recommendation(user, product)
    # Combine the results from all methods
    combined_results = list(chain(item_based_collaborative_filtering_results, user_based_collaborative_filtering_results, content_based_recommendation_results))

    # Filter out duplicate items
    unique_results = list(set(combined_results))

    return unique_results







def product_detail(request, category_slug, slug):
    product = get_object_or_404(Product, slug=slug)
    UserItemInteraction.objects.create(user=request.user, product=product)
    recommended_products = hybrid_recommendation(request.user, product)
    return render(request, 'product_detail.html', {
        'product': product,
        'recommended_products' : recommended_products,
    })






def category_detail(request, slug):
    category = get_object_or_404(Category, slug=slug)
    products = category.products.all()
    return render(request, 'category_detail.html', {
        'category' : category,
        'products' : products,
        })








def vendor_detail(request, pk):
    user = User.objects.get(pk=pk)
    products = user.products.all()
    return render(request, 'vendor_detail.html', {
        'user' : user,
        'products' : products,
        })







def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('home')  # Replace 'home' with the name of your home page URL pattern
        else:
            # Handle invalid login credentials
            error_message = "Invalid username or password. Please try again."
            return render(request, 'login.html', {'error_message': error_message})
    
    return render(request, 'login.html')





@login_required
def logout_view(request):
    logout(request)
    return redirect('home')  # Replace 'home' with the name of your home page URL pattern