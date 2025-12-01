import warnings
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import load_model

# Suppress warnings
warnings.filterwarnings("ignore")

def train_model():
    """
    This function trains the model from scratch and saves it.
    """
    print("--- Starting Model Training ---")
    
    # 1. Data Loading and Preprocessing
    print("Setting up data generators...")
    train_dir = "dogs-vs-cats/train"
    
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    print("✅ Data generators created.")

    # 2. Model Definition
    print("Defining the model...")
    base_model = MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    print("✅ Model defined.")

    # 3. Model Compilation
    print("Compiling the model...")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ Model compiled.")

    # 4. Model Training
    print("Starting model training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5
    )
    print("✅ Model training complete.")

    # 5. Save the Trained Model
    print("Saving the trained model...")
    model.save("catvsdog_trained_new.h5")
    print("✅ New trained model saved as 'catvsdog_trained_new.h5'.")

def predict_single_image(model, image_path):
    """
    Predicts the class of a single image.
    """
    print(f"--- Predicting image: {image_path} ---")
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        img_normalized = img_resized.astype('float32') / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)
        
        prediction = model.predict(img_input)
        predicted_class = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
        confidence = prediction[0][0] if predicted_class == 'Dog' else 1.0 - prediction[0][0]
        
        print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")

        # Show the image
        plt.imshow(img_normalized)
        plt.title(f"Predicted: {predicted_class}")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Prediction error: {e}")

def main():
    """
    Main function to run the script.
    """
    # Load the pre-trained model
    try:
        model = load_model('catvsdog_fixed.h5')
        print("✅ Pre-trained model 'catvsdog_fixed.h5' loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading pre-trained model: {e}")
        print("Please make sure the model 'catvsdog_fixed.h5' is in the same directory.")
        return

    # Example of predicting a single image
    # You can change the path to any image you want to test.
    # predict_single_image(model, 'dogs-vs-cats/test/cats/cat.10007.jpg')

    # Example of predicting a random image from the test set
    print("\n--- Predicting a random image from the test set ---")
    test_dir = r"dogs-vs-cats/test"
    cat_dir = os.path.join(test_dir, 'cats')
    dog_dir = os.path.join(test_dir, 'dogs')

    if os.path.exists(cat_dir) and os.path.exists(dog_dir):
        category = random.choice(['cat', 'dog'])
        img_dir = cat_dir if category == 'cat' else dog_dir
        if os.listdir(img_dir):
            img_name = random.choice(os.listdir(img_dir))
            img_path = os.path.join(img_dir, img_name)
            predict_single_image(model, img_path)
        else:
            print(f"No images found in {img_dir}")
    else:
        print("Test directories for cats and dogs not found.")


if __name__ == '__main__':
    # By default, this script will run the prediction on a random image.
    # To train the model, you can uncomment the following line:
    # train_model()
    
    main()
