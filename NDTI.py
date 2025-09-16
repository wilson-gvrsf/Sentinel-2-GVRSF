"""
NDTI (Normalized Difference Tillage Index) Calculator with Advanced Masking
Combines Sentinel-2 NDTI processing with comprehensive masking:
- Cloud masking (clouds, shadows, cirrus)
- Land type masking (cropland areas only) 
- Quality filtering for best results

NDTI Formula: (B11 - B12) / (B11 + B12)
- B11: SWIR1 band (1610nm) - Shortwave Infrared 1
- B12: SWIR2 band (2190nm) - Shortwave Infrared 2

NDTI is sensitive to crop residue and tillage practices:
- Higher values (0.1 to 0.4): More crop residue/recent tillage
- Lower values (-0.1 to -0.4): Less residue, more exposed soil
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize Google Earth Engine
PROJECT_ID = '590577866979'  # Replace with your Google Cloud Project ID

try:
    ee.Initialize(project=PROJECT_ID)
    print("Google Earth Engine initialized successfully!")
except Exception as e:
    print(f"Error initializing GEE: {e}")
    print("Please ensure you have proper authentication and project setup")

class LandTypeMask:
    """
    Land Type Masking using ESA WorldCover dataset
    Filters analysis to cropland areas only (class 40)
    """
    def __init__(self, EE_initialized=True):
        if not EE_initialized:
            ee.Initialize(project=PROJECT_ID)
        
        # Load ESA WorldCover dataset (2020-2021 data)
        self.worldcover = ee.ImageCollection('ESA/WorldCover/v200')
        
        # ESA WorldCover class mapping - we only care about cropland (40)
        self.class_mapping = {
            10: 'Tree cover',
            20: 'Shrubland', 
            30: 'Grassland',
            40: 'Cropland',  # ← TARGET CLASS for agricultural analysis
            50: 'Built-up',
            60: 'Bare/sparse vegetation',
            70: 'Snow and ice',
            80: 'Permanent water bodies',
            90: 'Herbaceous wetland',
            95: 'Mangroves',
            100: 'Moss and lichen'
        }

    def get_cropland_mask(self, geometry):
        """
        Create cropland-only mask for the specified geometry
        Returns binary mask: 1 = cropland, 0 = other land types
        """
        # Get most recent WorldCover image
        #CHIMA COMMMENT: This is probably fine, because normally cropland is going to stay cropland throughout a year. 
        #                However a more robust way to do this would be either taken the mode value, or the most recent image
        worldcover_image = self.worldcover.first().clip(geometry)
        
        # Create binary cropland mask (1 = cropland, 0 = everything else)
        cropland_mask = worldcover_image.eq(40).rename('cropland_mask')
        
        return cropland_mask

def get_aoi_geometry(lon, lat, box_width, box_height):
    """
    Create Area of Interest (AOI) geometry from center point and dimensions
    
    Args:
        lon (float): Longitude of center point
        lat (float): Latitude of center point  
        box_width (int): Width in meters
        box_height (int): Height in meters
    
    Returns:
        ee.Geometry.Rectangle: Bounding box geometry
    """
    # Convert meters to degrees (latitude-dependent for longitude)
    width_deg = box_width / (111320 * np.cos(np.radians(lat)))
    height_deg = box_height / 111132
    
    return ee.Geometry.Rectangle([
        lon - width_deg/2, lat - height_deg/2,
        lon + width_deg/2, lat + height_deg/2
    ])

class NDTIProcessor:
    """
    NDTI (Normalized Difference Tillage Index) Processor with Advanced Masking
    """
    
    def __init__(self, location, box_size, years, verbose=True):
        """
        Initialize NDTI processor
        
        Args:
            location (list): [longitude, latitude] center coordinates
            box_size (list): [width_meters, height_meters] 
            years (list): Years for analysis ['2024'] or ['2023', '2024']
            verbose (bool): Print processing details
        """
        self.location = location
        self.box_size = box_size  
        self.years = years
        self.verbose = verbose
        
        # Create geometry for area of interest
        self.aoi_geometry = get_aoi_geometry(location[0], location[1], 
                                           box_size[0], box_size[1])
        
        # Initialize land type masking
        self.land_mask = LandTypeMask(EE_initialized=True)
        
        # Get cropland mask for this region
        self.cropland_mask = self.land_mask.get_cropland_mask(self.aoi_geometry)
        
        # Set up date range
        if isinstance(years, list) and len(years) > 1:
            start_year, end_year = int(years[0]), int(years[-1])
        else:
            start_year = end_year = int(years[0]) if isinstance(years, list) else int(years)
            
        self.date_range = [f'{start_year}-01-01', f'{end_year}-12-31']
        
        if self.verbose:
            print(f"📍 AOI Center: {location[1]:.2f}°N, {location[0]:.2f}°E")
            print(f"📐 AOI Size: {box_size[0]/1000:.1f}km x {box_size[1]/1000:.1f}km")
            print(f"📅 Date Range: {self.date_range[0]} to {self.date_range[1]}")

    def calculate_ndti(self, image):
        """
        Calculate NDTI from Sentinel-2 SWIR bands
        
        NDTI Formula: (B11 - B12) / (B11 + B12)
        
        Where:
        - B11: SWIR1 (1610nm) - Shortwave Infrared band 1
        - B12: SWIR2 (2190nm) - Shortwave Infrared band 2
        
        The NDTI leverages the spectral differences between these SWIR bands:
        - Crop residue has higher reflectance in B11 than B12
        - Bare soil shows less difference between B11 and B12
        - This creates contrast for detecting tillage and residue patterns
        
        Args:
            image (ee.Image): Sentinel-2 image
            
        Returns:
            ee.Image: Original image with NDTI band added
        """
        # Extract SWIR bands for NDTI calculation
        b11 = image.select('B11')  # SWIR1 (1610nm) - sensitive to crop residue
        b12 = image.select('B12')  # SWIR2 (2190nm) - less sensitive to residue
        
        # Calculate NDTI: (B11 - B12) / (B11 + B12)
        # This ratio highlights differences in SWIR reflectance
        # Higher values = more crop residue (recent tillage)
        # Lower values = less residue (older tillage or bare soil)
        ndti = (b11.subtract(b12)
                  .divide(b11.add(b12))
                  .rename('NDTI'))
        
        return image.addBands(ndti)

    def apply_advanced_cloud_masking(self, image, qa_band='cs_cdf', clear_threshold=0.80):
        """
        Apply comprehensive cloud and quality masking
        
        Uses multiple masking layers:
        1. Cloud Score+ for probabilistic cloud detection
        2. Scene Classification Layer (SCL) for categorical masking
        3. Removes clouds, shadows, water, snow, and low-quality pixels
        
        Args:
            image (ee.Image): Sentinel-2 image with Cloud Score+ data
            qa_band (str): Cloud Score+ quality band 
            clear_threshold (float): Minimum clear sky probability (0-1)
            
        Returns:
            ee.Image: Masked image with only high-quality pixels
        """
        # 1. Cloud Score+ probabilistic masking
        cs = image.select(qa_band)
        cloud_mask = cs.gte(clear_threshold)  # Keep pixels >= threshold probability of clear sky
        
        # 2. Scene Classification Layer (SCL) categorical masking  
        scl = image.select('SCL')
        
        # Create masks for unwanted pixel types
        cloud_shadow_mask = scl.neq(3)   # Remove cloud shadows
        cloud_medium_mask = scl.neq(8)   # Remove medium probability clouds
        cloud_high_mask = scl.neq(9)     # Remove high probability clouds  
        cirrus_mask = scl.neq(10)        # Remove thin cirrus clouds
        snow_mask = scl.neq(11)          # Remove snow/ice
        water_mask = scl.neq(6)          # Remove water bodies
        saturated_mask = scl.neq(1)      # Remove saturated pixels
        dark_mask = scl.neq(2)           # Remove dark area pixels
        
        # 3. Combine all quality masks
        combined_mask = (cloud_mask
                        .And(cloud_shadow_mask)
                        .And(cloud_medium_mask)
                        .And(cloud_high_mask) 
                        .And(cirrus_mask)
                        .And(snow_mask)
                        .And(water_mask)
                        .And(saturated_mask)
                        .And(dark_mask))
        
        return image.updateMask(combined_mask)

    def apply_cropland_mask(self, image, sentinel_resolution=10):
        """
        Apply cropland land-type mask to restrict analysis to agricultural areas
        
        Args:
            image (ee.Image): Sentinel-2 image
            sentinel_resolution (int): Resampling resolution in meters
            
        Returns:
            ee.Image: Image masked to cropland areas only
        """
        #CHIMA COMMMENT: 1) GEE has it's own built in reprojecting that it applies when combining 2 seperate images. I've noticed some
        #                issues when reprojecting manually as done here, sometimes.
        #                2) the resolution of the B11 and B12 bands is 20meters NOT 10m as the B4 band, which you use to reproject the cropland data
        #                So actually this reprojection is a waste, because GEE has to reproject again to 20m when used on the B11 and B12 banse
        #                3) You can see detail about the Sentinel-2 data (like band resolution and wavelength) from here: 
        #                   https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/
        # Reproject cropland mask to match Sentinel-2 resolution for efficiency
        cropland_reprojected = self.cropland_mask.reproject(
            crs=image.select('B4').projection(), 
            scale=sentinel_resolution
        )
        
        # Apply mask: keep only cropland pixels (value = 1)
        cropland_valid = cropland_reprojected.eq(1)
        
        return image.updateMask(cropland_valid)

    def calculate_valid_pixel_count(self, image):
        """
        Count valid pixels after all masking for quality assessment
        
        Args:
            image (ee.Image): Masked image
            
        Returns:
            ee.Image: Image with valid_pixel_count property added
        """
        # Use B4 (red band) mask as representative of overall image mask
        mask = image.select('B4').mask().unmask(0)
        
        # Count valid pixels in the AOI
        count_dict = mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=self.aoi_geometry,
            scale=10,  # 10m resolution
            maxPixels=1e9,
            bestEffort=True
        )
        #CHIMA COMMMENT: Makes sense to work with 20m resolution here since that's the res of B11 and B12 bands
        
        count = count_dict.values().get(0)
        return image.set('valid_pixel_count', count)

    def process_sentinel2_collection(self, clear_threshold=0.80, min_valid_pixels=100):
        """
        Process Sentinel-2 collection with comprehensive masking and NDTI calculation
        
        Processing pipeline:
        1. Load Sentinel-2 Surface Reflectance and Cloud Score+ data
        2. Filter by date, area, and initial cloud cover
        3. Apply advanced cloud masking 
        4. Apply cropland land-type masking
        5. Calculate NDTI spectral index
        6. Count valid pixels and filter low-quality images
        
        Args:
            clear_threshold (float): Cloud Score+ clear sky threshold (0-1)
            min_valid_pixels (int): Minimum valid pixels to keep image
            
        Returns:
            ee.ImageCollection: Processed collection with NDTI bands
        """
        
        if self.verbose:
            print("\n🛰️  PROCESSING SENTINEL-2 DATA FOR NDTI ANALYSIS")
            print("="*60)
        
        # 1. Load Sentinel-2 Surface Reflectance and Cloud Score+ collections
        if self.verbose:
            print("1️⃣  Loading Sentinel-2 Surface Reflectance and Cloud Score+ data...")
            
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        cs_plus_collection = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

        #CHIMA COMMMENT: Not necessary at all, but it'd be nice to see the number of valid images before filtering, just to identify if filtering might
        #              be too agressive
        
        # 2. Initial filtering by date, area, and cloud cover
        if self.verbose:
            print("2️⃣  Filtering by date, area, and cloud cover...")
            
        filtered_s2 = (s2_collection
                      .filterDate(self.date_range[0], self.date_range[1])
                      .filterBounds(self.aoi_geometry)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))  # Pre-filter high cloud images
        #CHIMA COMMMENT: Think the 'CLOUDY_PIXEL_PERCENTAGE' filter is uncessary since you already filter when an image is covered by 80% clouds
        #               if you want more strict cloud filtering then just change 'clear_threshold' to a lower value rather than adding more code
        
        initial_count = filtered_s2.size().getInfo()
        if self.verbose:
            print(f"   📊 Images after initial filtering: {initial_count}")
        
        if initial_count == 0:
            print("❌ No images found after initial filtering!")
            return ee.ImageCollection([])
        
        # 3. Link with Cloud Score+ and apply advanced masking
        if self.verbose:
            print("3️⃣  Applying advanced cloud and shadow masking...")
        #CHIMA COMMMENT: Misprint. You're next step is applying ALL masking (cloud, shadow, and landtype). As well as calculating the NDTI
        
        def apply_all_masks(image):
            """Apply cloud masking, cropland masking, and NDTI calculation"""
            # Cloud masking
            masked_image = self.apply_advanced_cloud_masking(image, 'cs_cdf', clear_threshold)
            # Cropland masking  
            cropland_masked = self.apply_cropland_mask(masked_image)
            # NDTI calculation
            ndti_image = self.calculate_ndti(cropland_masked)
            # Pixel count calculation
            final_image = self.calculate_valid_pixel_count(ndti_image)
            
            return final_image
        
        # Link collections and apply all processing
        processed_collection = (filtered_s2
                               .linkCollection(cs_plus_collection, ['cs_cdf'])
                               .map(apply_all_masks))
        
        # 4. Filter by valid pixel count to ensure quality
        if self.verbose:
            print("4️⃣  Filtering images by valid pixel count...")
            
        final_collection = processed_collection.filter(
            ee.Filter.gt('valid_pixel_count', min_valid_pixels)
        ) #CHIMA COMMMENT: Might make more sense to have 'min_valid_pixels' be realtive to how many total pixels there could be
          #                That way in case there's a lot of missing pixels do to the crop region being small, you aren't removing 
          #                useful data. You'd have to count the total pixels of the first image after just applying the landtype
          #                and aoi mask for this
      
        final_count = final_collection.size().getInfo()
        removed_count = initial_count - final_count
        
        if self.verbose:
            print(f"   📊 Removed {removed_count} low-quality images")
            print(f"   📊 Final high-quality images: {final_count}")
            
        if final_count == 0:
            print("⚠️  Warning: No images remain after quality filtering!")
            print("   Try lowering clear_threshold or min_valid_pixels parameters")
            return ee.ImageCollection([])
        
        return final_collection

    def calculate_ndti_statistics(self, collection, seasonal_analysis=True):
        """
        Calculate comprehensive NDTI statistics
        
        Args:
            collection (ee.ImageCollection): Processed image collection with NDTI
            seasonal_analysis (bool): Whether to calculate seasonal statistics
            
        Returns:
            dict: NDTI statistics including annual and seasonal results
        """
        
        if self.verbose:
            print("\n📊 CALCULATING NDTI STATISTICS")
            print("="*40)
        
        results = {'annual': {}, 'seasonal': {}}
        
        # Annual NDTI statistics
        if self.verbose:
            print("📈 Calculating annual NDTI statistics...")
            
        # Calculate median NDTI across all images (reduces noise from outliers)
        annual_median_ndti = collection.select('NDTI').median()
        
        # Calculate comprehensive statistics over the AOI
        annual_stats = annual_median_ndti.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.minMax(), 
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.percentile([25, 50, 75]),
                sharedInputs=True
            ),
            geometry=self.aoi_geometry,
            scale=10,  # 10m resolution for detailed analysis
            maxPixels=1e9
        ).getInfo()
        
        # Store annual results
        results['annual'] = {
            'image_count': collection.size().getInfo(),
            'ndti_mean': annual_stats.get('NDTI_mean'),
            'ndti_std': annual_stats.get('NDTI_stdDev'),
            'ndti_min': annual_stats.get('NDTI_min'),
            'ndti_max': annual_stats.get('NDTI_max'),
            'ndti_p25': annual_stats.get('NDTI_p25'),
            'ndti_median': annual_stats.get('NDTI_p50'),
            'ndti_p75': annual_stats.get('NDTI_p75')
        }
        
        if self.verbose:
            print(f"   ✅ Annual NDTI Mean: {results['annual']['ndti_mean']:.4f}")
            print(f"   📊 Annual NDTI Range: {results['annual']['ndti_min']:.4f} to {results['annual']['ndti_max']:.4f}")
        
        # Seasonal analysis if requested
        if seasonal_analysis:
            if self.verbose:
                print("🌱 Calculating seasonal NDTI statistics...")
                
            # Define seasons (Northern Hemisphere focused - adjust for region if needed)
            seasons = {
                'Spring': ['03-01', '05-31'],  # March-May: Spring planting
                'Summer': ['06-01', '08-31'],  # June-August: Growing season  
                'Fall': ['09-01', '11-30'],    # September-November: Harvest
                'Winter': ['12-01', '02-28']   # December-February: Post-harvest
            }
            
            year = int(self.years[0]) if isinstance(self.years, list) else int(self.years)
            
            for season_name, (start_month_day, end_month_day) in seasons.items():
                # Handle winter season crossing year boundary
                if season_name == 'Winter':
                    start_date = f'{year}-{start_month_day}'
                    end_date = f'{year+1}-{end_month_day}'
                else:
                    start_date = f'{year}-{start_month_day}'
                    end_date = f'{year}-{end_month_day}'
                
                # Filter collection to seasonal date range
                seasonal_collection = collection.filterDate(start_date, end_date)
                seasonal_count = seasonal_collection.size().getInfo()
                
                if seasonal_count > 0:
                    # Calculate seasonal median NDTI
                    seasonal_median = seasonal_collection.select('NDTI').median()
                    
                    # Calculate seasonal statistics
                    seasonal_stats = seasonal_median.reduceRegion(
                        reducer=ee.Reducer.mean().combine(
                            reducer2=ee.Reducer.stdDev(),
                            sharedInputs=True
                        ),
                        geometry=self.aoi_geometry,
                        scale=10,
                        maxPixels=1e9
                    ).getInfo()
                    
                    results['seasonal'][season_name] = {
                        'image_count': seasonal_count,
                        'ndti_mean': seasonal_stats.get('NDTI_mean'),
                        'ndti_std': seasonal_stats.get('NDTI_stdDev')
                    }
                    
                    if self.verbose:
                        print(f"   🌿 {season_name}: {results['seasonal'][season_name]['ndti_mean']:.4f} "
                              f"({seasonal_count} images)")
                else:
                    if self.verbose:
                        print(f"   ❌ {season_name}: No images available")
                    results['seasonal'][season_name] = None
        
        return results

def analyze_region_ndti(region_name, location, box_size, years):
    """
    Analyze NDTI for a single region with comprehensive output
    
    Args:
        region_name (str): Name for display purposes
        location (list): [longitude, latitude] coordinates  
        box_size (list): [width_meters, height_meters]
        years (list): Years for analysis
        
    Returns:
        dict: Complete NDTI analysis results
    """
    
    print(f"\n{'='*70}")
    print(f"🌾 NDTI ANALYSIS: {region_name}")
    print(f"{'='*70}")
    print(f"📍 Location: {location[1]:.2f}°N, {location[0]:.2f}°E")
    print(f"📐 Area: {box_size[0]/1000:.1f}km × {box_size[1]/1000:.1f}km")
    print(f"📅 Years: {years}")
    
    try:
        # Initialize NDTI processor
        processor = NDTIProcessor(
            location=location,
            box_size=box_size, 
            years=years,
            verbose=True
        )
        
        # Process Sentinel-2 data with all masking
        collection = processor.process_sentinel2_collection()
        
        if collection.size().getInfo() == 0:
            print(f"❌ No valid data for {region_name}")
            return None
        
        # Calculate NDTI statistics
        statistics = processor.calculate_ndti_statistics(collection, seasonal_analysis=True)
        
        # Prepare return data
        result = {
            'region_name': region_name,
            'location': location,
            'box_size': box_size,
            'years': years,
            'processor': processor,
            'collection': collection,
            'statistics': statistics
        }
        
        print(f"✅ {region_name} NDTI analysis completed successfully!")
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing {region_name}: {e}")
        return None

def main_ndti_analysis():
    """
    Main function to run NDTI analysis on three agricultural regions
    """
    
    print("🌍 MULTI-REGION NDTI (TILLAGE INDEX) ANALYSIS")
    print("="*80)
    print("Analyzing tillage patterns using NDTI in three major agricultural regions")
    print("NDTI = (B11 - B12) / (B11 + B12) using Sentinel-2 SWIR bands")
    print("="*80)
    
    # Define study regions
    regions = {
        'Southeast Asia (Vietnam)': {
            'location': [105.8, 10.8],  # Ho Chi Minh City area - rice agriculture
            'box_size': [50000, 50000],  # 50km x 50km
            'years': ['2024']
        },
        'Saskatchewan, Canada': {
            'location': [-106.6, 52.1],  # Saskatoon area - wheat/canola agriculture
            'box_size': [50000, 50000],  # 50km x 50km  
            'years': ['2024']
        },
        'Europe (Northern France)': {
            'location': [2.3, 49.5],  # Northern France - diverse agriculture
            'box_size': [50000, 50000],  # 50km x 50km
            'years': ['2024']
        }
    }
    
    # Store all results
    all_results = []
    
    # Analyze each region
    for region_name, params in regions.items():
        result = analyze_region_ndti(
            region_name=region_name,
            location=params['location'],
            box_size=params['box_size'],
            years=params['years']
        )
        
        if result is not None:
            all_results.append(result)
    
    # Display comprehensive results summary
    if all_results:
        print(f"\n{'='*80}")
        print(f"📋 NDTI ANALYSIS SUMMARY - {len(all_results)} REGIONS")
        print(f"{'='*80}")
        
        # Create results DataFrame for easy comparison
        summary_data = []
        
        for result in all_results:
            stats = result['statistics']['annual']
            region = result['region_name']
            
            summary_data.append({
                'Region': region,
                'Images': stats['image_count'],
                'NDTI_Mean': stats['ndti_mean'],
                'NDTI_Std': stats['ndti_std'], 
                'NDTI_Min': stats['ndti_min'],
                'NDTI_Max': stats['ndti_max'],
                'NDTI_P25': stats['ndti_p25'],
                'NDTI_Median': stats['ndti_median'],
                'NDTI_P75': stats['ndti_p75']
            })
        
        # Display results table
        df_summary = pd.DataFrame(summary_data)
        print("📊 ANNUAL NDTI STATISTICS")
        print("-" * 80)
        print(df_summary.to_string(index=False, float_format='%.4f'))
        
        # Display seasonal results if available
        print(f"\n📅 SEASONAL NDTI BREAKDOWN")
        print("-" * 50)
        
        seasonal_data = []
        for result in all_results:
            region = result['region_name']
            seasonal_stats = result['statistics']['seasonal']
            
            for season, stats in seasonal_stats.items():
                if stats is not None:
                    seasonal_data.append({
                        'Region': region,
                        'Season': season,
                        'Images': stats['image_count'],
                        'NDTI_Mean': stats['ndti_mean'],
                        'NDTI_Std': stats['ndti_std']
                    })
        
        if seasonal_data:
            df_seasonal = pd.DataFrame(seasonal_data)
            print(df_seasonal.to_string(index=False, float_format='%.4f'))
        
        # NDTI Interpretation Guide
        print(f"\n{'='*80}")
        print("🔍 NDTI INTERPRETATION GUIDE")
        print("="*80)
        print("NDTI (Normalized Difference Tillage Index) measures tillage intensity:")
        print("")
        print("🌾 HIGH NDTI VALUES (0.1 to 0.4):")
        print("   • More crop residue present")
        print("   • Recent tillage activity") 
        print("   • Conservation tillage practices")
        print("   • Stubble or straw left on fields")
        print("")
        print("🏺 LOW NDTI VALUES (-0.1 to -0.4):")
        print("   • Less crop residue")
        print("   • Intensive tillage (clean tillage)")
        print("   • More exposed bare soil")
        print("   • Conventional tillage practices")
        print("")
        print("🌱 MEDIUM NDTI VALUES (around 0):")
        print("   • Mixed tillage conditions")
        print("   • Partial residue cover")
        print("   • Transitional agricultural practices")
        print("")
        print("📊 SPECTRAL PHYSICS:")
        print("   • B11 (1610nm): Higher reflectance from crop residue")
        print("   • B12 (2190nm): Less sensitive to residue")
        print("   • Ratio emphasizes residue vs. soil spectral differences")
        print("")
        print("🌍 REGIONAL CONTEXT:")
        print("   • Vietnam: Rice agriculture - expect lower NDTI (flooded fields)")
        print("   • Saskatchewan: Grain crops - expect variable NDTI (tillage diversity)")  
        print("   • France: Mixed crops - expect moderate NDTI (European practices)")
        
        return all_results
    
    else:
        print("❌ No regions were successfully analyzed!")
        return None

if __name__ == "__main__":
    try:
        # Run the comprehensive NDTI analysis
        results = main_ndti_analysis()
        
        if results:
            print(f"\n✨ NDTI analysis completed for {len(results)} regions!")
            print("📁 Results stored in 'results' variable for further analysis")
            
            # Optional: Save results to CSV
            # You can uncomment these lines to save results
            # summary_df = pd.DataFrame([...])  # Create DataFrame from results
            # summary_df.to_csv('ndti_analysis_results.csv', index=False)
            
        else:
            print("❌ NDTI analysis failed for all regions!")
            
    except Exception as e:
        print(f"❌ Critical error in NDTI processing: {e}")

        print("Ensure Google Earth Engine authentication and project setup are correct")
