import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

import shutil, os


stam = [
    '/home/borisef/data/cat/images/IMG_20210627_225110.jpg',
    '/home/borisef/data/cat/images/IMG_20210705_084125__01.jpg',
    '/home/borisef/data/cat/images/IMG_20210713_213907.jpg',
    '/home/borisef/data/cat/images/IMG_20210716_183123-02.jpeg',
    '/home/borisef/data/cat/images/IMG_20210718_213435.jpg',
    '/home/borisef/data/cat/images/IMG_20211226_111130.jpg',
    '/home/borisef/data/cat/images/IMG_20211226_145210.jpg',
    '/home/borisef/data/cat/images/IMG_20211226_145213.jpg',
    '/home/borisef/data/cat/images/IMG_20211229_114002.jpg',
    '/home/borisef/data/cat/images/IMG_20211229_114004.jpg',
    '/home/borisef/data/cat/images/IMG_20220101_102512.jpg',
    '/home/borisef/data/cat/images/IMG_20220102_145054.jpg',
    '/home/borisef/data/cat/images/IMG_20220102_145059.jpg'
]

df = pd.DataFrame({
    'image_path': stam,
    'ground_truth_class': [0,0,0, 1, 1, 1, 1, 1,2,2,2,2,2],
    'predicted_class': [1,2,0, 1, 1, 0, 2, 1,2,2,2,0,1]
})


max_images_to_present = 500  # Set your desired maximum number of images to present

labels = ['cat', 'dog', 'car']
save_images_dir = '/home/borisef/temp/cm'

def compute_confusion_matrix_data(df):
    return pd.crosstab(df['ground_truth_class'], df['predicted_class']).values.tolist()


def generate_html_page_w(df, matrix_data, max_images, labels, num_gt_samples_per_class):
    # Generate HTML for displaying confusion matrix
    html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Confusion Matrix</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h2>Confusion Matrix</h2>
            <div id="confusion-matrix" style="width: 500px; height: 500px;"></div>
            <div id="image-content"></div>
            <script>
                // Function to update image content
                function updateImageContent(imagesHtml, message) {{
                    document.getElementById('image-content').innerHTML = '<h2>' + message + '</h2>' + imagesHtml;
                }}

                // Add confusion matrix plot to the HTML page
                var matrixData = {str(matrix_data).replace('[', '[').replace(']', ']')};
                var labels = {str(labels)};

                var data = {{
                    z: matrixData,
                    type: 'heatmap',
                    colorscale: 'Viridis'
                }};

                var layout = {{
                    xaxis: {{title: 'Predicted Class', tickvals: [0, 1, 2], ticktext: labels}},
                    yaxis: {{title: 'Actual Class', tickvals: [0, 1, 2], ticktext: labels}}
                }};

                Plotly.newPlot('confusion-matrix', [data], layout);

                // Add click event listener to the confusion matrix cells
                document.getElementById('confusion-matrix').on('plotly_click', function(data) {{
                    var cell_i = labels[data.points[0].y];
                    var cell_j = labels[data.points[0].x];
                    var cell_images = {df.to_dict(orient='records')};
                    cell_images = cell_images.filter(function (item) {{
                        return labels[item.ground_truth_class] == cell_i && labels[item.predicted_class] == cell_j;
                    }}).map(function (item) {{
                        return item.image_path;
                    }});

                    // Display images directly in the main window
                    var imagesHtml = "";
                    var totalImages = cell_images.length;

                    if ({max_images} < totalImages) {{
                        // Randomly shuffle the array of images
                        for (var i = totalImages - 1; i > 0; i--) {{
                            var j = Math.floor(Math.random() * (i + 1));
                            var temp = cell_images[i];
                            cell_images[i] = cell_images[j];
                            cell_images[j] = temp;
                        }}
                    }}

                    var imagesToDisplay = Math.min({max_images}, totalImages);
                    for (var k = 0; k < imagesToDisplay; k++) {{
                        imagesHtml += '<img src="' + cell_images[k] + '" alt="Image" style="width:200px; margin: 5px;">\\n';
                    }}

                    var message = 'Images of class ' + cell_i + ', classified as ' + cell_j + '. Total such images: ' + totalImages + '. Total images of class ' + cell_i + ': ' + {num_gt_samples_per_class}[labels.indexOf(cell_i)];
                    updateImageContent(imagesHtml, message);
                }});
            </script>
        </body>
        </html>
    '''

    # Save the HTML content to a file
    with open('confusion_matrix.html', 'w') as html_file:
        html_file.write(html_content)

# Example usage

def generate_html_page(df, matrix_data, max_images, labels, num_gt_samples_per_class):
    # Compute the maximum value in the confusion matrix
    max_value = max(map(max, matrix_data))

    # Generate HTML for displaying confusion matrix
    html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Confusion Matrix</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h2>Confusion Matrix</h2>
            <div id="confusion-matrix" style="width: 500px; height: 500px;"></div>
            <div id="image-content"></div>
            <script>
                // Function to update image content
                function updateImageContent(imagesHtml, message) {{
                    document.getElementById('image-content').innerHTML = '<h2>' + message + '</h2>' + imagesHtml;
                }}

                // Add confusion matrix plot to the HTML page
                var matrixData = {str(matrix_data).replace('[', '[').replace(']', ']')};
                var labels = {str(labels)};

                var data = {{
                    z: matrixData,
                    type: 'heatmap',
                    colorscale: 'Viridis',
                    zmin: 0,  // Set the minimum value for the color scale
                    zmax: {max_value},  // Set the maximum value for the color scale
                }};

                var layout = {{
                    xaxis: {{title: 'Predicted Class', tickvals: [0, 1, 2], ticktext: labels}},
                    yaxis: {{title: 'Actual Class', tickvals: [0, 1, 2], ticktext: labels}}
                }};

                Plotly.newPlot('confusion-matrix', [data], layout);

                // Add click event listener to the confusion matrix cells
                document.getElementById('confusion-matrix').on('plotly_click', function(data) {{
                    var cell_i = labels[data.points[0].y];
                    var cell_j = labels[data.points[0].x];
                    var cell_images = {df.to_dict(orient='records')};
                    cell_images = cell_images.filter(function (item) {{
                        return labels[item.ground_truth_class] == cell_i && labels[item.predicted_class] == cell_j;
                    }}).map(function (item) {{
                        return item.image_path;
                    }});

                    // Display images directly in the main window
                    var imagesHtml = "";
                    var totalImages = cell_images.length;

                    if ({max_images} < totalImages) {{
                        // Randomly shuffle the array of images
                        for (var i = totalImages - 1; i > 0; i--) {{
                            var j = Math.floor(Math.random() * (i + 1));
                            var temp = cell_images[i];
                            cell_images[i] = cell_images[j];
                            cell_images[j] = temp;
                        }}
                    }}

                    var imagesToDisplay = Math.min({max_images}, totalImages);
                    for (var k = 0; k < imagesToDisplay; k++) {{
                        imagesHtml += '<img src="' + cell_images[k] + '" alt="Image" style="width:200px; margin: 5px;">\\n';
                    }}

                    var message = 'Images of class ' + cell_i + ', classified as ' + cell_j + '. Total such images: ' + totalImages + '. Total images of class ' + cell_i + ': ' + {num_gt_samples_per_class}[labels.indexOf(cell_i)];
                    updateImageContent(imagesHtml, message);
                }});
            </script>
        </body>
        </html>
    '''

    # Save the HTML content to a file
    with open('confusion_matrix.html', 'w') as html_file:
        html_file.write(html_content)


def copy_images_to_directory(df, save_images_dir):
    if save_images_dir is not None:
        for index, row in df.iterrows():
            source_path = row['image_path']
            cell_i = labels[row['ground_truth_class']]
            cell_j = labels[row['predicted_class']]
            dir_path = os.path.join(save_images_dir, f"{cell_i}_{cell_j}")

            # Create the directory if it doesn't exist
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # Copy the image to the corresponding folder
            shutil.copy(source_path, dir_path)

# Call the function to generate HTML page with confusion matrix
confusion_matrix_data = compute_confusion_matrix_data(df)

num_gt_samples_per_class = list(np.array(np.array(confusion_matrix_data)).sum(axis=1))

#generate_html_page(df, confusion_matrix_data, max_images_to_present, labels, save_images_dir, num_gt_samples_per_class)
generate_html_page(df, confusion_matrix_data, max_images_to_present, labels, num_gt_samples_per_class)

# Function to copy images to the corresponding folder

# Call the function to copy images to the corresponding folders
copy_images_to_directory(df, save_images_dir)