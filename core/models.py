from django.db import models
from django.contrib.auth.models import User
from statistics import mode 
from django.core.files import File 
from io import BytesIO
from PIL import Image 




class Userprofile(models.Model):
	user = models.OneToOneField(User, related_name='userprofile', on_delete=models.CASCADE)
	is_vendor = models.BooleanField(default=False)
	def __str__(self):
		return self.user.username 




class Category(models.Model):
	title = models.CharField(max_length=50)
	slug = models.SlugField(max_length=50)

	class Meta:
		verbose_name_plural = 'Categories'

	def __str__(self):
		return self.title




class Product(models.Model):
	DRAFT = 'draft'
	WAITING_APPROVAL = 'waitingapproval' 
	ACTIVE = 'active'
	DELETED = 'deleted'

	STATUS_CHOICES = {
		(DRAFT, 'Draft'), 
		(WAITING_APPROVAL, 'Waiting approval'),
		(ACTIVE, 'Active'),
		(DELETED, 'Deleted'),
	}
	user = models.ForeignKey(User, related_name='products', on_delete=models.CASCADE)
	category = models.ForeignKey(Category, related_name='products', on_delete=models.CASCADE)
	title = models.CharField(max_length=50)
	slug = models.SlugField(max_length=50)
	description = models.TextField(blank=True)
	tags = models.TextField(default='')
	price = models.PositiveIntegerField()
	image = models.ImageField(upload_to='uploads/product_images', blank=True, null=True)
	thumbnail = models.ImageField(upload_to='uploads/product_images/thumbnail', blank=True, null=True)
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)
	status = models.CharField(max_length=50, choices=STATUS_CHOICES, default=ACTIVE)
	average_rating = models.DecimalField(max_digits=3, decimal_places=1, default=0)
	view_count = models.PositiveIntegerField(default=0)

	class Meta:
		ordering = ('-created_at',)

	def __str__(self):
		return self.title


	def get_thumbnail(self):
		if self.thumbnail:
			return self.thumbnail.url 
		else:
			if self.image:
			 	self.thumbnail = self.make_thumbnail(self.image)
			 	self.save()

			 	return self.thumbnail.url 

			else:
				return 'https://via.placeholder.com/240*240*.jpg'

	def make_thumbnail(self, image, size=(300, 300)):
		img = Image.open(image)
		img.convert('RGB')
		img.thumbnail(size)

		thumb_io = BytesIO()
		img.save(thumb_io, 'JPEG', quantity=85)
		name = image.name.replace('uploads/product_images/', '')
		thumbnail = File(thumb_io, name=name)

		return thumbnail






class UserItemInteraction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
    	return f"{self.user.username} viewed {self.product.title}"





class Order(models.Model):
	first_name = models.CharField(max_length=255)
	last_name = models.CharField(max_length=255)
	address = models.CharField(max_length=255)
	zipcode = models.CharField(max_length=255)
	city = models.CharField(max_length=255)
	payment_intent = models.CharField(max_length=255, blank=True, null=True, default=None)
	paid_amount = models.IntegerField(blank=True, null=True)
	is_paid = models.BooleanField(default=False)
	created_by = models.ForeignKey(User, related_name='orders', on_delete=models.SET_NULL, null=True)
	created_at = models.DateTimeField(auto_now_add=True)






class UserPurchase(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    order = models.ForeignKey('Order', related_name='purchases', on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} bought {self.product.title}"






class UserRating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    rating = models.DecimalField(max_digits=3, decimal_places=2)






class OrderItem(models.Model):
	order = models.ForeignKey(Order, related_name='items', on_delete=models.CASCADE)
	product = models.ForeignKey(Product, related_name='items', on_delete=models.CASCADE)
	price = models.IntegerField()
	quantity = models.IntegerField(default=1)

	def __str__(self):
		return self.product





class Comment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Comment by {self.user.username} on {self.product.title}"

