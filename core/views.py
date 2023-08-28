from django.conf import settings
from django.contrib import messages
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.utils.text import slugify
from django.db.models import Q, Count, Avg
from .models import Product, Category, UserItemInteraction, Order, OrderItem, Userprofile, UserPurchase, Comment, UserRating
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
from django.http import JsonResponse, HttpResponse
from django.template.loader import render_to_string
from .cart import Cart
from .forms import OrderForm, ProductForm, CommentForm, RatingForm
import stripe



def home(request):
    products = Product.objects.filter(status=Product.ACTIVE)
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












@login_required
def mystore(request):
    products = request.user.products.exclude(status=Product.DELETED)
    order_items = OrderItem.objects.filter(product__user=request.user)

    return render(request, 'mystore.html', {
        'products' : products,
        'order_items' : order_items,
        })





@login_required
def mystore_order_detail(request, pk):
    order = get_object_or_404(Order, pk=pk)

    return render(request, 'mystore_order_detail.html', {
        'order' : order,
        })



'''
@login_required
def add_product(request):
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            product = form.save(commit=False)
            product.user = request.user
            product.save()
            return redirect('mystore')  # Redirect to a page displaying the list of products
    else:
        form = ProductForm()
    return render(request, 'add_product.html', {'form': form})
'''




@login_required
def add_product(request):
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)

        if form.is_valid():
            title = request.POST.get('title')
            product = form.save(commit=False)
            product.user = request.user
            product.slug = slugify(title)
            product.save()
            messages.success(request, 'Product added successfully!')
            return redirect('mystore')

    else:
        form = ProductForm()

    return render(request, 'add_product.html', {
        'title' : 'Add product',
        'form' : form,
        })






@login_required
def edit_product(request, pk):
    product = Product.objects.filter(user=request.user).get(pk=pk)
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES, instance=product)

        if form.is_valid():
            form.save()
            messages.success(request, 'Changes saved successfully!')
            return redirect('mystore')

    else:
        form = ProductForm(instance=product)

    return render(request, 'add_product.html', {
        'title' : 'Edit product',
        'product' : product,
        'form' : form
        })







@login_required
def delete_product(request, pk):
    product = Product.objects.filter(user=request.user).get(pk=pk)
    product.status = Product.DELETED 
    product.save()
    messages.success(request, 'Product deleted successfully')
    return redirect('mystore')






@login_required
def myaccount(request):
    return render(request, 'myaccount.html')





def add_to_cart(request, product_id):
    cart = Cart(request)
    cart.add(product_id)

    return redirect('cart_view')




def success(request):
    return render(request, 'success.html', {
        'name' : request.user.username
        })




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
            UserPurchase.objects.create(user=request.user, product=product, order=order)

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
    products = Product.objects.filter(status=Product.ACTIVE).filter(Q(tags__icontains=query))
    return render(request, 'search.html', {
        'query' : query,
        'products' : products,
        })








import nltk
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token] 
    tokens = ['NUM' if token.isdigit() else token for token in tokens]
    tokens = [token.strip() for token in tokens]
    tokens = list(set(tokens))
    return tokens

    






def item_based_collaborative_filtering(product):
    products = Product.objects.filter(category=product.category)
    # Create a dictionary to store product descriptions for each product
    product_tags = {}
    product_tags[product.slug] = product.tags
    for prod in products:
        product_slug = prod.slug
        product_tag = prod.tags
        if product_slug not in product_tags:
            product_tags[product_slug] = product_tag

    tfidf_vectorizer = TfidfVectorizer()

    product_slugs = []
    product_texts = []
    for product_slug, tags in product_tags.items():
        tag = preprocess_text(tags)
        product_texts.append(" ".join(tag))
        product_slugs.append(product_slug)

    tfidf_matrix = tfidf_vectorizer.fit_transform(product_texts)
    target_product_index = product_slugs.index(product.slug)
    cosine_similarities = cosine_similarity(tfidf_matrix[target_product_index], tfidf_matrix)

    # Get similarity scores for the target product
    similarity_scores = cosine_similarities[0]

    # Create a list of tuples containing product slug and similarity score
    product_similarity = [(product_slug, similarity) for product_slug, similarity in zip(product_slugs, similarity_scores)]

    # Sort the list based on similarity scores in descending order
    sorted_product_similarity = sorted(product_similarity, key=lambda x: x[1], reverse=True)

    # Exclude the target product slug from the recommended products
    recommended_products = [get_object_or_404(Product, slug=slug) for slug, _ in sorted_product_similarity if slug != product.slug]

    # Return the top 5 recommended products
    return recommended_products[:5]








def content_based_recommendation(user, product):
    viewed_interactions = UserItemInteraction.objects.filter(user=user)
    viewed_product_ids = list(set(viewed_interactions.values_list('product__id', flat=True)))
    viewed_products = list(set(interaction.product for interaction in viewed_interactions))

    user_profile = ' '.join(f'{product.tags}' for product in viewed_products)
    all_products = Product.objects.all()
    product_profiles = [' '.join([str(product.tags)]) for product in all_products]
    
    profiles = []
    user_profile = preprocess_text(user_profile)
    profiles.append(" ".join(user_profile))

    for x in product_profiles:
        x = preprocess_text(x)
        profiles.append(" ".join(x))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(profiles)
    cosine_similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    # Create a list of tuples with product ID and similarity score
    product_similarity_pairs = [(all_products[i].id, cosine_similarities[0][i]) for i in range(len(all_products)) if all_products[i].id not in viewed_product_ids]
    
    # Sort the list of product-similarity pairs by similarity score in descending order
    sorted_product_similarity_pairs = sorted(product_similarity_pairs, key=lambda x: x[1], reverse=True)
    
    # Get recommended product IDs
    recommended_product_ids = [product_id for product_id, _ in sorted_product_similarity_pairs]
    
    # Retrieve recommended products
    recommended_products = [get_object_or_404(Product, id=x) for x in recommended_product_ids if x != product.id][:5]
    return recommended_products
    







def user_based_collaborative_filtering(target_user):
    # Separate interactions of the target user
    target_user_interactions = UserItemInteraction.objects.filter(user=target_user)

    # Get interactions of all other users
    all_other_users_interactions = UserItemInteraction.objects.exclude(user=target_user)

    # Create a dictionary to store product views for each user
    products_viewed_by_user = {}
    for interaction in all_other_users_interactions:
        user_id = interaction.user.id
        product_id = interaction.product.id
        if user_id not in products_viewed_by_user:
            products_viewed_by_user[user_id] = set()
        products_viewed_by_user[user_id].add(product_id)

    # Convert product views to text for TF-IDF vectorization
    user_product_text = []
    user_ids = []
    for user_id, product_ids in products_viewed_by_user.items():
        product_text = " ".join(map(str, product_ids))
        user_product_text.append(product_text)
        user_ids.append(user_id)

    # Add target user's interactions to the beginning of the list
    target_product_text = " ".join(map(str, [interaction.product.id for interaction in target_user_interactions]))
    user_product_text.insert(0, target_product_text)
    user_ids.insert(0, target_user.id)

    # Create TF-IDF vectorizer and compute TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(user_product_text)

    cosine_similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix)[0]

    # Find indices of top 5 similar users (excluding the target user)
    similar_user_indices = cosine_similarities.argsort()[-6:-1][::-1]

    # Get products viewed by top 5 similar users but not by the target user
    top_products = set()
    for user_index in similar_user_indices:
        user_id = user_ids[user_index]
        top_products.update(products_viewed_by_user[user_id])

    target_user_viewed_products = set([interaction.product.id for interaction in target_user_interactions])
    top_products -= target_user_viewed_products

    # Create a list of tuples (product_id, view_count) for the top products
    top_product_views = [(product_id, Product.objects.get(id=product_id).view_count) for product_id in top_products]

    # Sort products by view count in descending order
    top_product_views.sort(key=lambda x: x[1], reverse=True)

    # Get the top 5 products
    top_5_products = [get_object_or_404(Product, id=product_id) for product_id, _ in top_product_views[:5]]

    return top_5_products








'''
def hybrid_recommendation(user, product):
    item_based_collaborative_filtering(product)
    user_based_collaborative_filtering(user)
    content_based_recommendation(user, product)
    # Combine the results from all methods
    #combined_results = list(chain(item_based_collaborative_filtering_results, user_based_collaborative_filtering_results, content_based_recommendation_results))

    # Filter out duplicate items
    #unique_results = list(set(combined_results))

    #return unique_results
'''







def product_detail(request, category_slug, slug):
    product = get_object_or_404(Product, slug=slug, status=Product.ACTIVE)
    product.view_count += 1
    product.save()
    UserItemInteraction.objects.create(user=request.user, product=product)
    recommended_products = item_based_collaborative_filtering(product)
    recommended_products2 = user_based_collaborative_filtering(request.user)
    recommended_products3 = content_based_recommendation(request.user, product)

    user_bought_products = UserPurchase.objects.filter(user=request.user, product=product).exists()

    comment_form = CommentForm(request.POST or None)
    rating_form = RatingForm(request.POST or None)  

    if request.method == 'POST' and comment_form.is_valid() and user_bought_products:
        comment_text = comment_form.cleaned_data['text']
        Comment.objects.create(user=request.user, product=product, text=comment_text)

    if request.method == 'POST' and rating_form.is_valid() and user_bought_products:
        rating = rating_form.cleaned_data['rating']
        UserRating.objects.update_or_create(user=request.user, product=product, defaults={'rating': rating})


    comments = Comment.objects.filter(product=product)
    ratings = UserRating.objects.filter(product=product)
    average_rating = ratings.aggregate(Avg('rating'))['rating__avg'] or 0
   
    no_full_stars = int(average_rating)
    full_stars = [1 for x in range(no_full_stars)]
    half_star = average_rating - no_full_stars
    no_empty_stars = 10 - no_full_stars - (half_star > 0)
    empty_stars = [1 for x in range(no_empty_stars)]



    return render(request, 'product_detail.html', {
        'product': product,
        'recommended_products': recommended_products,
        'recommended_products2': recommended_products2,
        'recommended_products3': recommended_products3,
        'user_bought_products': user_bought_products,
        'comment_form': comment_form,
        'comments' : comments,
        'rating_form': rating_form,
        'ratings': ratings,
        'average_rating': average_rating,
        'full_stars': full_stars, 'half_star': half_star, 'empty_stars': empty_stars,
    })








def category_detail(request, slug):
    category = get_object_or_404(Category, slug=slug)
    products = category.products.filter(status=Product.ACTIVE)
    return render(request, 'category_detail.html', {
        'category' : category,
        'products' : products,
        })









def vendor_detail(request, pk):
    user = get_object_or_404(User, pk=pk)
    products = user.products.filter(status=Product.ACTIVE)
    return render(request, 'vendor_detail.html', {
        'user': user,
        'products': products,
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