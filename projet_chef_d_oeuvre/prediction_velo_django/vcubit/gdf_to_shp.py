import os
import shutil
import tempfile

def gdf_to_shp_zip(gdf, name, shp_dir):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            gdf.to_file(os.path.join(temp_dir, name), driver='ESRI Shapefile', encoding='utf-8')
            shutil.make_archive(os.path.join(shp_dir, name), 'zip', os.path.join(temp_dir, name))
    except Exception as e:
        print(f"An error occured: {e}")
        return 1
    return 0
