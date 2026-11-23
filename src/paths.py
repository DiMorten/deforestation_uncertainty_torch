import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path(
    os.environ.get("DEFORESTATION_DATA_ROOT", Path(__file__).resolve().parents[1] / "datasets")
)


class Paths:
    def __init__(self, site, years, data_root=None, before_2008_filename=None):
        root = str(Path(data_root).resolve()) if data_root else str(DEFAULT_DATA_ROOT)

        self.reference_folder = os.path.join(root, "deforestation", site)
        self.optical_im_folder = os.path.join(root, "sentinel2", site)
        self.experiment = os.path.join(root, "deforestation", "experiments", site)

        self.hydrography = os.path.join(self.reference_folder, "hydgrography.tif")
        self.deforestation_past_years = os.path.join(self.reference_folder, "deforestation_past_years.tif")

        if before_2008_filename:
            self.deforestation_before_2008 = os.path.join(self.reference_folder, before_2008_filename)

        self.optical_im_past_dates = {
            year: os.path.join(self.optical_im_folder, str(year))
            for year in years
        }

        self.cloud_mask = {
            year: os.path.join(self.optical_im_folder, str(year), f"cloudmask_{site}_{year}.npy")
            for year in years
        }

        # deforestation_time excludes the most recent year in the original setup
        self.deforestation_time = {
            year: os.path.join(self.reference_folder, f"deforestation_time_normalized_{year}.npy")
            for year in years
        }


class PathsPA(Paths):
    def __init__(self, data_root=None):
        super().__init__(
            site="PA",
            years=[2019, 2018, 2017, 2016, 2015],
            data_root=data_root,
            before_2008_filename="deforestation_before_2008_PA.tif",
        )
        self.optical_im = os.path.join(self.optical_im_folder, "2019")

        self.im_filenames = {
            2020: [
                "S2_PA_2020_07_15_B1_B2_B3.tif",
                "S2_PA_2020_07_15_B4_B5_B6.tif",
                "S2_PA_2020_07_15_B7_B8_B8A.tif",
                "S2_PA_2020_07_15_B9_B10_B11.tif",
                "S2_PA_2020_07_15_B12.tif",
            ],
            2019: [
                "COPERNICUS_S2_20190721_20190726_B1_B2_B3.tif",
                "COPERNICUS_S2_20190721_20190726_B4_B5_B6.tif",
                "COPERNICUS_S2_20190721_20190726_B7_B8_B8A.tif",
                "COPERNICUS_S2_20190721_20190726_B9_B10_B11.tif",
                "COPERNICUS_S2_20190721_20190726_B12.tif",
            ],
            2018: [
                "COPERNICUS_S2_20180721_20180726_B1_B2_B3.tif",
                "COPERNICUS_S2_20180721_20180726_B4_B5_B6.tif",
                "COPERNICUS_S2_20180721_20180726_B7_B8_B8A.tif",
                "COPERNICUS_S2_20180721_20180726_B9_B10_B11.tif",
                "COPERNICUS_S2_20180721_20180726_B12.tif",
            ],
            2015: [
                "PA_S2_2015_B1_B2_B3_crop.tif",
                "PA_S2_2015_B4_B5_B6_crop.tif",
                "PA_S2_2015_B7_B8_B8A_crop.tif",
                "PA_S2_2015_B9_B10_B11_crop.tif",
                "PA_S2_2015_B12_crop.tif",
            ],
        }

        root = str(Path(data_root).resolve()) if data_root else str(DEFAULT_DATA_ROOT)
        landsat_base = os.path.join(root, "landsat", "PA")
        self.landsat = [
            os.path.join(landsat_base, f"landsat_PA_{year}.tif")
            for year in [2018, 2019, 2020]
        ]


class PathsMT(Paths):
    def __init__(self, data_root=None):
        super().__init__(
            site="MT",
            years=[2020, 2019, 2018, 2017, 2016],
            data_root=data_root,
            before_2008_filename="deforestation_before_2008_MT.tif",
        )
        del self.deforestation_time[2020]
        self.optical_im = os.path.join(self.optical_im_folder, "2020")

        self.im_filenames = {
            2020: [
                "S2_R1_MT_2020_08_03_2020_08_15_B1_B2.tif",
                "S2_R1_MT_2020_08_03_2020_08_15_B3_B4.tif",
                "S2_R1_MT_2020_08_03_2020_08_15_B5_B6.tif",
                "S2_R1_MT_2020_08_03_2020_08_15_B7_B8.tif",
                "S2_R1_MT_2020_08_03_2020_08_15_B8A_B9.tif",
                "S2_R1_MT_2020_08_03_2020_08_15_B10_B11.tif",
                "S2_R1_MT_2020_08_03_2020_08_15_B12.tif",
            ],
            2019: [
                "S2_R1_MT_2019_08_02_2019_08_05_B1_B2.tif",
                "S2_R1_MT_2019_08_02_2019_08_05_B3_B4.tif",
                "S2_R1_MT_2019_08_02_2019_08_05_B5_B6.tif",
                "S2_R1_MT_2019_08_02_2019_08_05_B7_B8.tif",
                "S2_R1_MT_2019_08_02_2019_08_05_B8A_B9.tif",
                "S2_R1_MT_2019_08_02_2019_08_05_B10_B11.tif",
                "S2_R1_MT_2019_08_02_2019_08_05_B12.tif",
            ],
            2018: [
                "MT_S2_07_26_28_31_2018_B1_B2_crop.tif",
                "MT_S2_07_26_28_31_2018_B3_B4_crop.tif",
                "MT_S2_07_26_28_31_2018_B5_B6_crop.tif",
                "MT_S2_07_26_28_31_2018_B7_B8_crop.tif",
                "MT_S2_07_26_28_31_2018_B8A_B9_crop.tif",
                "MT_S2_07_26_28_31_2018_B10_B11_crop.tif",
                "MT_S2_07_26_28_31_2018_B12_crop.tif",
            ],
            2017: [
                "MT_S2_07_26_28_2017_B1_B2_crop.tif",
                "MT_S2_07_26_28_2017_B3_B4_crop.tif",
                "MT_S2_07_26_28_2017_B5_B6_crop.tif",
                "MT_S2_07_26_28_2017_B7_B8_crop.tif",
                "MT_S2_07_26_28_2017_B8A_B9_crop.tif",
                "MT_S2_07_26_28_2017_B10_B11_crop.tif",
                "MT_S2_07_26_28_2017_B12_crop.tif",
            ],
            2016: [
                "MT_S2_2016_07_21_08_07_B1_B2_crop.tif",
                "MT_S2_2016_07_21_08_07_B3_B4_crop.tif",
                "MT_S2_2016_07_21_08_07_B5_B6_crop.tif",
                "MT_S2_2016_07_21_08_07_B7_B8_crop.tif",
                "MT_S2_2016_07_21_08_07_B8A_B9_crop.tif",
                "MT_S2_2016_07_21_08_07_B10_B11_crop.tif",
                "MT_S2_2016_07_21_08_07_B12_crop.tif",
            ],
        }

        root = str(Path(data_root).resolve()) if data_root else str(DEFAULT_DATA_ROOT)
        landsat_base = os.path.join(root, "landsat", "MT")
        self.landsat = [
            os.path.join(landsat_base, f"landsat_MT_{year}.tif")
            for year in [2019, 2020, 2021]
        ]


class PathsMS(Paths):
    def __init__(self, data_root=None):
        super().__init__(
            site="MS",
            years=[2020, 2019],
            data_root=data_root,
            before_2008_filename="deforestation_before_2000_MS.tif",
        )
        self.im_filenames = {
            2019: ["merged_2019_crop.tif"],
            2020: ["merged_2020_crop.tif"],
        }


class PathsPI(Paths):
    def __init__(self, data_root=None):
        super().__init__(
            site="PI",
            years=[2020, 2019],
            data_root=data_root,
            before_2008_filename="deforestation_before_2000_PI.tif",
        )
        self.im_filenames = {
            2019: ["merged_2019_crop.tif"],
            2020: ["merged_2020_crop.tif"],
        }


class PathsMA(Paths):
    def __init__(self, data_root=None):
        super().__init__(
            site="MA",
            years=[2021, 2020],
            data_root=data_root,
            before_2008_filename=None,
        )
        bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
        self.im_filenames = {
            2020: [f"T23KQV_20200602T130251_{b}.jp2" for b in bands],
            2021: [f"T23KQV_20210811T130249_{b}.jp2" for b in bands],
        }