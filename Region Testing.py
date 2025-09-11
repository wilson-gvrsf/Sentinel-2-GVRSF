"""
Multi-Region Cropland Analysis with Advanced Masking
Analyzes cropland in Southeast Asia, Saskatchewan Canada, and Europe
Combines Sentinel-2 processing with ESA WorldCover land type classification
"""

import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gzip
import io
import requests
from PIL import Image

# Initialize Google Earth Engine
PROJECT_ID = '590577866979'  # Replace with your Google Cloud Project ID

try:
    ee.Initialize(project=PROJECT_ID)
    print("Google Earth Engine initialized successfully!")
except Exception as e:
    print(f"Error initializing GEE: {e}")
    print("Please ensure you have:")
    print("1. Run 'earthengine authenticate'")
    print("2. Created a Google Cloud Project")
    print("3. Enabled Earth Engine API")
    print("4. Set the correct PROJECT_ID in the code")

class LandType:
    """A class to get land type using ESA WorldCover dataset. 
    (Data is only valid from 2020 to 2021)   
    """
    def __init__(self, GEE_project_id='tlg-erosion1', DataRes=0.00009, EE_initialized=True):
        if not EE_initialized: 
            # Initialize Earth Engine
            ee.Authenticate()
            ee.Initialize(project=GEE_project_id)
        
        # Load ESA WorldCover dataset
        worldcover = ee.ImageCollection('ESA/WorldCover/v200')
        self.worldcover = worldcover
        self.DataRes = DataRes

        # ESA WorldCover class mapping
        self.class_mapping = {
            10: 'Tree cover',
            20: 'Shrubland',
            30: 'Grassland',
            40: 'Cropland',  # Wilson is only interested in cropland - flag 40
            50: 'Built-up',
            60: 'Bare/sparse vegetation',
            70: 'Snow and ice',
            80: 'Permanent water bodies',
            90: 'Herbaceous wetland',
            95: 'Mangroves',
            100: 'Moss and lichen'
        }

    def get_land_cover_for_region(self, Geometry):
        """Get land cover data for specified geometry"""
        # Get the most recent WorldCover image
        worldcover_image = self.worldcover.first()
        
        # Clip to the area of interest
        clipped = worldcover_image.clip(Geometry)
        
        return {'image': clipped}

    def Map_LandType(self, landcover_image):
        """Create simplified land type map focusing on cropland"""
        # Create a mask where only cropland (40) is valid (1), others are invalid (0)
        cropland_mask = landcover_image.eq(40).rename('cropland_mask')
        
        return cropland_mask

def get_aoi(lon, lat, box_width, box_height):
    """Create area of interest geometry"""
    # Convert box dimensions from meters to degrees (approximate)
    # Using more accurate conversion factors for latitude-dependent longitude
    width_deg = box_width / (111320 * np.cos(np.radians(lat)))  # latitude-dependent longitude conversion
    height_deg = box_height / 110540  # meters to degrees latitude
    
    return ee.Geometry.Rectangle([
        lon - width_deg/2, lat - height_deg/2,
        lon + width_deg/2, lat + height_deg/2
    ])

class CroplandProcessor:
    """Cropland Processing with Advanced Masking - Works for any region"""
    
    def __init__(self, Location, Box, Years, Verbose=True, GEE_project_id='tlg-erosion1', 
                 SentRes=10, ShowPlots=True, plot_scale=100):
        """
        Initialize the cropland processor
        
        Args:
            Location (list): [longitude, latitude] of center point
            Box (list): [width_m, height_m] dimensions in meters
            Years (list): ['start_year', 'end_year'] or single year
            Verbose (bool): Print detailed information
            SentRes (int): Sentinel-2 resolution in meters
            ShowPlots (bool): Whether to generate visualizations
            plot_scale (int): Scale for plotting
        """
        
        # Initialize Earth Engine (assuming already done)
        self.LT = LandType(EE_initialized=True)
        
        # Get the geometry of the area of interest
        self.AoI_geom = get_aoi(Location[0], Location[1], Box[0], Box[1])
        
        # Get land cover data and create cropland mask
        result = self.LT.get_land_cover_for_region(Geometry=self.AoI_geom)
        self.RegionMap = self.LT.Map_LandType(result['image'])
        
        # Set up date ranges
        if isinstance(Years, list) and len(Years) > 1:
            year0, yearE = int(Years[0]), int(Years[-1])
        else:
            year0 = yearE = int(Years[0]) if isinstance(Years, list) else int(Years)
            
        self.Dates = [f'{year0}-01-01', f'{yearE}-12-31']
        
        # Store parameters
        self.Location = Location
        self.Box = Box
        self.Years = Years
        self.Verbose = Verbose
        self.SentRes = SentRes
        self.ShowPlots = ShowPlots
        self.plot_scale = plot_scale

    def Pull_Process_Sentinel_data(self, QA_BAND='cs_cdf', CLEAR_THRESHOLD=0.80):
        """
        Process Sentinel-2 data with comprehensive masking
        
        Args:
            QA_BAND (str): Cloud Score+ quality band ('cs_cdf' recommended)
            CLEAR_THRESHOLD (float): Minimum clear sky probability (0-1)
        
        Returns:
            ee.ImageCollection: Processed image collection with valid_pixel_count property
        """
        
        # Support functions for processing
        def mask_clouds_advanced(img):
            """Enhanced cloud masking with shadows, snow, and water removal"""
            # Get the Cloud Score+ data
            cs = img.select(QA_BAND)
            
            # Basic cloud mask
            cloud_mask = cs.gte(CLEAR_THRESHOLD)
            
            # Get Scene Classification Layer (SCL) for additional masking
            scl = img.select('SCL')
            
            # Create masks for various unwanted pixels
            cloud_shadow_mask = scl.neq(3)  # Remove cloud shadows
            cloud_medium_mask = scl.neq(8)  # Remove medium probability clouds
            cloud_high_mask = scl.neq(9)    # Remove high probability clouds
            cirrus_mask = scl.neq(10)       # Remove thin cirrus
            snow_mask = scl.neq(11)         # Remove snow/ice
            water_mask = scl.neq(6)         # Remove water bodies
            saturated_mask = scl.neq(1)     # Remove saturated pixels
            dark_mask = scl.neq(2)          # Remove dark area pixels
            
            # Combine all masks
            combined_mask = (cloud_mask
                            .And(cloud_shadow_mask)
                            .And(cloud_medium_mask) 
                            .And(cloud_high_mask)
                            .And(cirrus_mask)
                            .And(snow_mask)
                            .And(water_mask)
                            .And(saturated_mask)
                            .And(dark_mask))
            
            return img.updateMask(combined_mask)

        def set_pixel_count(image):
            """Calculate valid pixel count and add as property"""
            mask = image.select('B4').mask().unmask(0)
            count_dict = mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=self.AoI_geom,
                scale=self.SentRes,
                maxPixels=1e9,
                bestEffort=True)
            count = count_dict.values().get(0)
            return image.set('valid_pixel_count', count)
        
        def apply_landtype_mask(image):
            """Apply cropland mask - only keep cropland areas"""
            # Reproject the landtype mask to match Sentinel-2 resolution for efficiency
            landtype_mask = self.RegionMap.reproject(crs=image.select('B4').projection(), scale=self.SentRes)
            landtype_valid_mask = landtype_mask.eq(1)  # Only cropland areas
            return image.updateMask(landtype_valid_mask)

        # 1) Load Sentinel-2 and Cloud Score+ collections
        print("1.) Loading Sentinel-2, Surface Reflectance, and Cloud Score+ collections...")
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

        # Initial filtering by date and area
        filtered_s2_date_area = (s2
            .filterBounds(self.AoI_geom)
            .filterDate(self.Dates[0], self.Dates[1]))
        
        total_count = filtered_s2_date_area.size().getInfo()
        print(f"Total images in date range: {total_count}")
        
        # Add cloud cover filtering to reduce dataset size
        filtered_s2_date_area = filtered_s2_date_area.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        
        print(f"Filtering from {self.Dates[0]} to {self.Dates[1]}")
        initial_count = filtered_s2_date_area.size().getInfo()
        print(f"Images after cloud pre-filter (<30% clouds): {initial_count}")
        
        if initial_count == 0:
            print("❌ No images found after initial filtering!")
            return ee.ImageCollection([])

        # 2) Apply cloud masking
        print("2) Filtering pixels blocked by clouds and shadows...")
        filtered_s2 = (filtered_s2_date_area
            .linkCollection(csPlus, [QA_BAND])
            .map(mask_clouds_advanced))

        # 3) Apply land type mask (cropland only)
        print("3) Filtering to cropland areas only...")
        print("   (This step may take a moment for land type classification...)")
        land_masked_collection = filtered_s2.map(apply_landtype_mask)

        # 4) Calculate pixel counts and filter
        print("4) Calculating valid pixel counts and removing low-quality images...")
        land_masked_collection_with_counts = land_masked_collection.map(set_pixel_count)
        
        # Filter to keep only images with sufficient valid pixels
        final_s2 = land_masked_collection_with_counts.filter(
            ee.Filter.gt('valid_pixel_count', 100))
        
        clean_count = final_s2.size().getInfo()
        removed = initial_count - clean_count
        print(f"Removed {removed} images with insufficient valid pixels")
        print(f"Final remaining images: {clean_count}")
        
        if clean_count == 0:
            print("⚠️ Warning: No images remain after all filtering!")
            return ee.ImageCollection([])

        # Return collection WITH counts for best image selection
        return final_s2

def create_rgb_composite(image):
    """Create RGB composite from Sentinel-2 bands (B4, B3, B2)"""
    rgb = image.select(['B4', 'B3', 'B2']).multiply(0.0001)
    return rgb

def create_false_color_composite(image):
    """Create false color composite (NIR, Red, Green) for vegetation analysis"""
    false_color = image.select(['B8', 'B4', 'B3']).multiply(0.0001)
    return false_color

def visualize_image(image, region, vis_params, title="Sentinel-2 Image"):
    """
    Visualize Earth Engine image using matplotlib with direct URL download approach
    """
    try:
        print(f"   ⏳ Processing {title}...")
        
        thumbnail_params = {
            'region': region,
            'dimensions': 800,
            'format': 'png'
        }
        thumbnail_params.update(vis_params)
        
        url = image.getThumbURL(thumbnail_params)
        print(f"   🔗 URL: {url}")
        
        try:
            print(f"   📥 Downloading image...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            img = Image.open(io.BytesIO(response.content))
            print(f"   📏 Image size: {img.size}")
            
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.title(f"{title}\nImage Size: {img.size}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            print(f"   ✅ Successfully displayed: {title}")
            return True
            
        except Exception as display_error:
            print(f"   ⚠️ Could not display image: {display_error}")
            print(f"   🌐 Copy this URL to your browser: {url}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error generating {title}: {e}")
        return False

def analyze_single_region(region_name, location, box_size, years):
    """
    Analyze cropland for a single region and return the best RGB image
    
    Args:
        region_name (str): Name of the region for display
        location (list): [longitude, latitude] coordinates
        box_size (list): [width, height] in meters
        years (list): Year(s) for analysis
    
    Returns:
        dict: Analysis results including RGB image
    """
    
    print(f"\n{'='*60}")
    print(f"🌾 ANALYZING: {region_name}")
    print(f"{'='*60}")
    print(f"📍 Location: {location[1]:.2f}°N, {location[0]:.2f}°E")
    print(f"📐 Area: {box_size[0]/1000:.1f}km x {box_size[1]/1000:.1f}km")
    print(f"📅 Analysis Period: {years}")
    
    try:
        # Initialize processor
        processor = CroplandProcessor(
            Location=location,
            Box=box_size,
            Years=years,
            Verbose=True,
            ShowPlots=False  # We'll handle plotting separately
        )
        
        # Process Sentinel-2 data
        print(f"\n🛰️ Processing Sentinel-2 data for {region_name}...")
        collection = processor.Pull_Process_Sentinel_data()
        
        if collection.size().getInfo() == 0:
            print(f"❌ No valid images found for {region_name}")
            return None
        
        # Get the best image
        print(f"\n🔍 Selecting best quality image for {region_name}...")
        best_image = collection.sort('valid_pixel_count', False).first()
        
        # Get quality metrics
        best_pixel_count = best_image.get('valid_pixel_count').getInfo()
        best_cloud_cover = best_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        best_scene_id = best_image.get('PRODUCT_ID').getInfo()
        
        print(f"🏆 Selected best image for {region_name}:")
        print(f"   📊 Valid cropland pixels: {best_pixel_count}")
        print(f"   ☁️ Cloud cover: {best_cloud_cover:.1f}%")
        print(f"   🛰️ Scene ID: {best_scene_id}")
        
        # Create RGB composite
        rgb_image = create_rgb_composite(best_image)
        
        # Get acquisition date
        acquisition_date = best_image.get('system:time_start').getInfo()
        formatted_date = datetime.fromtimestamp(acquisition_date/1000).strftime('%Y-%m-%d')
        
        return {
            'region_name': region_name,
            'processor': processor,
            'collection': collection,
            'rgb_image': rgb_image,
            'best_image': best_image,
            'pixel_count': best_pixel_count,
            'cloud_cover': best_cloud_cover,
            'scene_id': best_scene_id,
            'acquisition_date': formatted_date
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {region_name}: {e}")
        return None

def main_multi_region_analysis():
    """
    Main function to analyze cropland in three regions and display RGB composites
    """
    
    print("🌍 MULTI-REGION CROPLAND ANALYSIS")
    print("="*80)
    print("Analyzing cropland in Southeast Asia, Saskatchewan Canada, and Europe")
    print("="*80)
    
    # Define the three regions
    regions = {
        'Southeast Asia (Vietnam)': {
            'location': [105.8, 10.8],  # Ho Chi Minh City area
            'box_size': [50000, 50000],  # 50km x 50km
            'years': ['2024']
        },
        'Saskatchewan, Canada': {
            'location': [-106.6, 52.1],  # Saskatoon area - major agricultural region
            'box_size': [60000, 60000],  # 60km x 60km (larger for prairie agriculture)
            'years': ['2024']
        },
        'Europe (Northern France)': {
            'location': [2.3, 49.5],  # Northern France agricultural region
            'box_size': [50000, 50000],  # 50km x 50km
            'years': ['2024']
        }
    }
    
    # Store results for all regions
    all_results = []
    
    # Analyze each region
    for region_name, params in regions.items():
        result = analyze_single_region(
            region_name=region_name,
            location=params['location'],
            box_size=params['box_size'],
            years=params['years']
        )
        
        if result is not None:
            all_results.append(result)
            print(f"✅ {region_name} analysis completed successfully!")
        else:
            print(f"❌ {region_name} analysis failed!")
    
    # Display all RGB composites
    if all_results:
        print(f"\n🎨 DISPLAYING RGB COMPOSITES FOR ALL {len(all_results)} REGIONS")
        print("="*80)
        
        # Visualization parameters for RGB
        rgb_vis = {'min': 0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2']}
        
        for i, result in enumerate(all_results, 1):
            region_name = result['region_name']
            rgb_image = result['rgb_image']
            processor = result['processor']
            
            print(f"\n🌾 RGB COMPOSITE #{i}: {region_name}")
            print(f"📅 Acquisition Date: {result['acquisition_date']}")
            print(f"☁️ Cloud Cover: {result['cloud_cover']:.1f}%")
            print(f"📊 Valid Pixels: {result['pixel_count']}")
            
            title = f"{region_name} - Cropland RGB Composite"
            visualize_image(rgb_image, processor.AoI_geom, rgb_vis, title)
        
        # Summary
        print(f"\n📋 ANALYSIS SUMMARY")
        print("="*50)
        for result in all_results:
            print(f"🌾 {result['region_name']}:")
            print(f"   📅 Date: {result['acquisition_date']}")
            print(f"   ☁️ Cloud Cover: {result['cloud_cover']:.1f}%")
            print(f"   📊 Valid Cropland Pixels: {result['pixel_count']}")
            print()
        
        print(f"🎉 Successfully analyzed {len(all_results)} regions!")
        print("📊 All images show only cropland areas with comprehensive masking:")
        print("   ✅ Cloud masking (clouds, shadows, cirrus)")
        print("   ✅ Land type masking (cropland areas only)")
        print("   ✅ Quality filtering (best images selected)")
        
        return all_results
    
    else:
        print("❌ No regions were successfully analyzed!")
        return None

# Alternative function to test different regions easily
def analyze_custom_regions(custom_regions=None):
    """
    Analyze custom regions - allows easy modification of locations
    
    Args:
        custom_regions (dict): Dictionary of region definitions
                              Format: {'Region Name': {'location': [lon, lat], 
                                                      'box_size': [w, h], 
                                                      'years': [year]}}
    """
    
    if custom_regions is None:
        # Default three regions with different characteristics
        custom_regions = {
            'Mekong Delta, Vietnam': {
                'location': [105.8, 10.8],  # Rice-intensive region
                'box_size': [40000, 40000],
                'years': ['2024']
            },
            'Saskatchewan Prairies': {
                'location': [-107.0, 52.5],  # Wheat belt
                'box_size': [60000, 60000],
                'years': ['2024']
            },
            'Po Valley, Italy': {
                'location': [11.0, 45.0],  # European agricultural heartland
                'box_size': [50000, 50000],
                'years': ['2024']
            }
        }
    
    print(f"🌍 CUSTOM MULTI-REGION ANALYSIS")
    print("="*80)
    print(f"Analyzing {len(custom_regions)} custom regions")
    
    results = []
    
    for region_name, params in custom_regions.items():
        result = analyze_single_region(
            region_name=region_name,
            location=params['location'],
            box_size=params['box_size'],
            years=params['years']
        )
        
        if result is not None:
            results.append(result)
    
    # Display RGB composites for all successful regions
    if results:
        rgb_vis = {'min': 0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2']}
        
        print(f"\n🎨 DISPLAYING {len(results)} RGB COMPOSITES")
        print("="*60)
        
        for result in results:
            title = f"{result['region_name']} - Cropland RGB"
            visualize_image(result['rgb_image'], result['processor'].AoI_geom, 
                          rgb_vis, title)
    
    return results

if __name__ == "__main__":
    try:
        # Run the main multi-region analysis
        results = main_multi_region_analysis()
        
        # Uncomment the line below to test with custom regions instead
        # results = analyze_custom_regions()
        
        if results:
            print(f"\n✨ Multi-region cropland analysis completed!")
            print(f"📊 Successfully processed {len(results)} regions")
        else:
            print(f"\n❌ Multi-region analysis failed!")
            
    except Exception as e:
        print(f"❌ Error in multi-region processing: {e}")
        print("Make sure Google Earth Engine is properly authenticated and initialized")