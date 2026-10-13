"""Per-LoRA metadata used by the perceptual ground-truth pipeline.

``LORA_METADATA``  trigger words and the recommended LoRA weight per model.
``PROMPT_TEMPLATES`` generic prompt templates per LoRA category.
``LORA_TYPES``     category of each known LoRA (used to pick a template).
"""

LORA_METADATA = {
    # Action
    "action_SD1.5_EkuneSideDoggy": {
        "trigger_words": "sidedoggystyle, 1boy, ass, doggystyle, sex",
        "recommended_weight": 1.0
    },
    # Animal
    "animal_SD1.5_cat_20230627113759": {
        "trigger_words": "cat",
        "recommended_weight": 1.0
    },

    # Background
    "background_SD1.5_Ink scenery": {
        "trigger_words": "white background, scenery, ink, mountains, water, trees",
        "recommended_weight": 1.0
    },
    "background_SD1.5_model-lora-volcano-and-lava": {
        "trigger_words": "volcano, lava, fire, flames",
        "recommended_weight": 1.0
    },

    # Character
    "character_flux1s_wukong_convert": {
        "trigger_words": "wukong",
        "recommended_weight": 1.0
    },
    "character_flux1s_wukong": {
        "trigger_words": "wukong",
        "recommended_weight": 1.0,
        "rank": 16
    },
    "character_SD1.4_Eula-Model": {
        "trigger_words": "(eula, youla, /genshin impact/)",
        "recommended_weight": 1.0
    },
    "character_SD1.5_Sakuragi Hanamichi1": {
        "trigger_words": "Sakuragi Hanamichi1, red hair,short hair,pompadourshirt,buzz cut,basketball uniform",
        "recommended_weight": 0.7
    },
    "character_SD1.5_asianGirlsFace_v1": {
        "trigger_words": "FACE",
        "recommended_weight": 0.45
    },
    "character_SD1.5_koreanDollLikeness": {
        "trigger_words": "korean doll likeness",
        "recommended_weight": 0.5
    },
    "character_SD1.5_mix4": {"trigger_words": "MIX4",
        "recommended_weight": 0.55
    },
    
    # Clothing
    "clothing_SD1.5_basket_uniform-10": {
        "trigger_words": "wearing basket_uniform, inside basket field, back view, side view",
        "recommended_weight": 0.2
    },
    "clothing_SD1.5_equitation_outfit-10": {
        "trigger_words": "wearing equitation_outfit, wearing helmet, riding a horse, in front of a crowd",
        "recommended_weight": 0.35
    },
    "clothing_SD1.5_golf_outfit-10": {
        "trigger_words": "wearing golf_outfit, wearing golf_cap, golf field, golf cart background",
        "recommended_weight": 0.35
    },
    "clothing_SD1.5_hanfu_v30": {
        "trigger_words": "hanfu, ming style",
        "recommended_weight": 0.2
    },
    "clothing_SD1.5_Sexy Basketball Player": {
        "trigger_words": "basketballplayer, shirtless, shorts, dynamic movement, ball, sneakers",
        "recommended_weight": 0.5
    },
    "clothing_SD1.5_sexy_donghua clothes": {
        "trigger_words": "donghua_clothes, sexy_donghua",
        "recommended_weight": 1.0
    },
    "clothing_SD1.5_striped_dress-10": {
        "trigger_words": "wearing striped_dress",
        "recommended_weight": 0.3
    },

    # Concept
    "concept_XL_XLFantasyKnightsV4": {
        "trigger_words": "hkstyle",
        "recommended_weight": 1.0
    },

    # Quality
    "quality_SD1.5_add_detail": {
        "trigger_words": "add detail",
        "recommended_weight": 0.75
    },
    "quality_XL_add-detail-xl": {
        "trigger_words": "add detail",
        "recommended_weight": 0.75
    },

    # Style
    "style_flux1s_pixelart_schnell_v1_convert": {
        "trigger_words": "pixel art",
        "recommended_weight": 1.0
    },
    "style_flux1s_pixelart_schnell_v1": {
        "trigger_words": "pixel art",
        "recommended_weight": 1.0,
        "rank": 32
    },
    "style_SD1.5_3DMM_V12": {
        "trigger_words": "3DMM",
        "recommended_weight": 1.0
    },
    "style_SD1.5_afro_hairstyle": {
        "trigger_words": "afro_hairstyle",
        "recommended_weight": 0.25
    },
    "style_SD1.5_animeoutlineV4_16": {
        "trigger_words": "lineart, monochrome",
        "recommended_weight": 1.0
    },
    "style_SD1.5_Artehm_Chebokha-RTMCHBKH0123_Style": {
        "trigger_words": "Impasto, RTMCHBKH0123 Style",
        "recommended_weight": 0.9
    },

    "style_SD1.5_bichu-v0612": {
        "trigger_words": "bichu, oil painting, Impressionism",
        "recommended_weight": 0.55
    },
    "style_SD1.5_dreads_hairstyle": {
        "trigger_words": "dreads_hairstyle",
        "recommended_weight": 0.30
    },
    "style_SD1.5_hi_top_fade_hairstyle": {
        "trigger_words": "hi_top_fade_hairstyle",
        "recommended_weight": 0.30
    },
    "style_SD1.5_howlbgsv3": {
        "trigger_words": "in the style of  <lora:howllatentDADAP4:1>",
        "recommended_weight": 1.0
    },

    "style_SD1.5_MoXinV1": {"trigger_words": "shuimobysim, wuchangshuo, bonian, zhenbanqiao, badashanren",
        "recommended_weight": 0.65
    },
    "style_SD1.5_ruinedlol": {
        "trigger_words": "ruinedlol",
        "recommended_weight": 0.8
    },
    "style_SD1.5_tbh182-": {
        "trigger_words": "style of Clyde Caldwell",
        "recommended_weight": 1.0
    },
    "style_XL_CardboardStyleXL": {
        "trigger_words": "CardboardStyle",
        "recommended_weight": 1.0
    },
    "style_XL_ClassipeintXL2.1": {
        "trigger_words": "oil painting",
        "recommended_weight": 1.0
    },
    "style_XL_Cyberpunk _Anime_V2_light_version-SDXL": {
        "trigger_words": "anime, cyberpunk",
        "recommended_weight": 1.0
    },
    "style_XL_EldritchComicsXL1.2": {"trigger_words": "comic book",
        "recommended_weight": 1.0
    },
    "style_XL_glass": {
        "trigger_words": "Glass",
        "recommended_weight": 1.0
    },
    "style_XL_Lego_XL_v2.1": {"trigger_words": "LEGO Creator",
        "recommended_weight": 0.8
    },
    "style_XL_ral-dissolve-sdxl": {
        "trigger_words": "ral-dissolve",
        "recommended_weight": 1.0
    },
    "style_XL_Surreal Harmony": {
        "trigger_words": "Surreal Harmony",
        "recommended_weight": 1.0
    },
}


# Generic prompt templates per LoRA type; add or adjust as needed.
PROMPT_TEMPLATES = {
    "character": [
        "{trigger_words}, a portrait, masterpiece, best quality, detailed face",
        "{trigger_words}, 1girl, standing in a fantasy forest, cinematic lighting",
        "{trigger_words}, upper body, wearing a beautiful dress, simple background",
    ],
    "animal": [
        "{trigger_words}, a close-up portrait photo, masterpiece, best quality, detailed fur",
        "{trigger_words}, in its natural habitat, national geographic photo",
        "{trigger_words}, full body shot, simple background",
    ],
    "style": [
        "{trigger_words}, a majestic lion in the savannah",
        "{trigger_words}, a futuristic cyberpunk city with flying cars, neon lights",
        "{trigger_words}, a beautiful woman knight, fantasy art",
    ],
    "pose": [
        "{trigger_words}, 1girl, masterpiece, best quality, full body shot",
        "{trigger_words}, 1boy, detailed background",
    ],
    "clothing": [
        "{trigger_words}, 1girl, full body, high quality photo",
        "{trigger_words}, a model, fashion photography",
    ],
    "concept": [
        "{trigger_words}, a beautiful landscape, masterpiece, best quality",
        "{trigger_words}, a cinematic shot, epic scene",
    ],
    "quality": [
        "{trigger_words}, a portrait of a beautiful woman, masterpiece, best quality",
        "{trigger_words}, a majestic lion, detailed fur",
    ],
}

# Map each LoRA name to a prompt-template type.
LORA_TYPES = {
    # Action
    "action_SD1.5_EkuneSideDoggy": "pose",
    # Animal
    "animal_SD1.5_cat_20230627113759": "animal",
    # Background
    "background_SD1.5_Ink scenery": "concept",
    "background_SD1.5_model-lora-volcano-and-lava": "concept",
    # Character
    "character_flux1s_wukong_convert": "character",
    "character_SD1.4_Eula-Model": "character",
    "character_SD1.5_Sakuragi Hanamichi1": "character",
    # Clothing
    "clothing_SD1.5_basket_uniform-10": "clothing",
    "clothing_SD1.5_equitation_outfit-10": "clothing",
    "clothing_SD1.5_golf_outfit-10": "clothing",
    "clothing_SD1.5_hanfu_v30": "clothing",
    "clothing_SD1.5_Sexy Basketball Player": "clothing",
    "clothing_SD1.5_sexy_donghua clothes": "clothing",
    "clothing_SD1.5_striped_dress-10": "clothing",
    # Concept
    "concept_XL_XLFantasyKnightsV4": "concept",
    # Quality
    "quality_SD1.5_add_detail": "quality",
    "quality_XL_add-detail-xl": "quality",
    # Style
    "style_flux1s_pixelart_schnell_v1_convert": "style",
    "style_SD1.5_3DMM_V12": "style",
    "style_SD1.5_afro_hairstyle": "style",
    "style_SD1.5_animeoutlineV4_16": "style",
    "style_SD1.5_Artehm_Chebokha-RTMCHBKH0123_Style": "style",
    "style_SD1.5_asianGirlsFace_v1": "style",
    "style_SD1.5_bichu-v0612": "style",
    "style_SD1.5_dreads_hairstyle": "style",
    "style_SD1.5_hi_top_fade_hairstyle": "style",
    "style_SD1.5_howlbgsv3": "style",
    "style_SD1.5_koreanDollLikeness": "style",
    "style_SD1.5_mix4": "style",
    "style_SD1.5_MoXinV1": "style",
    "style_SD1.5_ruinedlol": "style",
    "style_SD1.5_tbh182-": "style",
    "style_XL_CardboardStyleXL": "style",
    "style_XL_ClassipeintXL2.1": "style",
    "style_XL_Cyberpunk _Anime_V2_light_version-SDXL": "style",
    "style_XL_EldritchComicsXL1.2": "style",
    "style_XL_glass": "style",
    "style_XL_Lego_XL_v2.1": "style",
    "style_XL_ral-dissolve-sdxl": "style",
    "style_XL_Surreal Harmony": "style",
}