from __future__ import annotations

from dataclasses import asdict

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column


# Define the models
class Base(DeclarativeBase, MappedAsDataclass):
    pass


class Item(Base):
    __tablename__ = "packages_all_staging"
    package_name: Mapped[str] = mapped_column()
    package_picture: Mapped[str] = mapped_column()
    url: Mapped[str] = mapped_column(primary_key=True)
    price: Mapped[float] = mapped_column()
    cash_discount: Mapped[float] = mapped_column()
    installment_month: Mapped[str] = mapped_column()
    price_after_cash_discount: Mapped[float] = mapped_column()
    installment_limit: Mapped[str] = mapped_column()
    price_to_reserve_for_this_package: Mapped[float] = mapped_column()
    shop_name: Mapped[str] = mapped_column()
    category: Mapped[str] = mapped_column()
    category_tags: Mapped[str] = mapped_column()
    preview_1_10: Mapped[str] = mapped_column()
    selling_point: Mapped[str] = mapped_column()
    meta_keywords: Mapped[str] = mapped_column()
    brand: Mapped[str] = mapped_column()
    min_max_age: Mapped[str] = mapped_column()
    locations: Mapped[str] = mapped_column()
    meta_description: Mapped[str] = mapped_column()
    price_details: Mapped[str] = mapped_column()
    package_details: Mapped[str] = mapped_column()
    important_info: Mapped[str] = mapped_column()
    payment_booking_info: Mapped[str] = mapped_column()
    general_info: Mapped[str] = mapped_column()
    early_signs_for_diagnosis: Mapped[str] = mapped_column()
    how_to_diagnose: Mapped[str] = mapped_column()
    hdcare_summary: Mapped[str] = mapped_column()
    common_question: Mapped[str] = mapped_column()
    know_this_disease: Mapped[str] = mapped_column()
    courses_of_action: Mapped[str] = mapped_column()
    signals_to_proceed_surgery: Mapped[str] = mapped_column()
    get_to_know_this_surgery: Mapped[str] = mapped_column()
    comparisons: Mapped[str] = mapped_column()
    getting_ready: Mapped[str] = mapped_column()
    recovery: Mapped[str] = mapped_column()
    side_effects: Mapped[str] = mapped_column()
    review_4_5_stars: Mapped[str] = mapped_column()
    brand_option_in_thai_name: Mapped[str] = mapped_column()
    brand_ranking_position: Mapped[int] = mapped_column()
    faq: Mapped[str] = mapped_column()
    embedding_package_name: Mapped[Vector] = mapped_column(Vector(1536))  # ada-002
    embedding_package_picture: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_url: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_installment_month: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_installment_limit: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_shop_name: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_category: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_category_tags: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_preview_1_10: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_selling_point: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_meta_keywords: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_brand: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_min_max_age: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_locations: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_meta_description: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_price_details: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_package_details: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_important_info: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_payment_booking_info: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_general_info: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_early_signs_for_diagnosis: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_how_to_diagnose: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_hdcare_summary: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_common_question: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_know_this_disease: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_courses_of_action: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_signals_to_proceed_surgery: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_get_to_know_this_surgery: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_comparisons: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_getting_ready: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_recovery: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_side_effects: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_review_4_5_stars: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_brand_option_in_thai_name: Mapped[Vector] = mapped_column(Vector(1536))
    embedding_faq: Mapped[Vector] = mapped_column(Vector(1536))

    def to_dict(self, include_embedding: bool = False):
        model_dict = asdict(self)
        if include_embedding:
            embedding_columns = [
                "embedding_package_name",
                "embedding_package_picture",
                "embedding_url",
                "embedding_installment_month",
                "embedding_installment_limit",
                "embedding_shop_name",
                "embedding_category",
                "embedding_category_tags",
                "embedding_preview_1_10",
                "embedding_selling_point",
                "embedding_meta_keywords",
                "embedding_brand",
                "embedding_min_max_age",
                "embedding_locations",
                "embedding_meta_description",
                "embedding_price_details",
                "embedding_package_details",
                "embedding_important_info",
                "embedding_payment_booking_info",
                "embedding_general_info",
                "embedding_early_signs_for_diagnosis",
                "embedding_how_to_diagnose",
                "embedding_hdcare_summary",
                "embedding_common_question",
                "embedding_know_this_disease",
                "embedding_courses_of_action",
                "embedding_signals_to_proceed_surgery",
                "embedding_get_to_know_this_surgery",
                "embedding_comparisons",
                "embedding_getting_ready",
                "embedding_recovery",
                "embedding_side_effects",
                "embedding_review_4_5_stars",
                "embedding_brand_option_in_thai_name",
                "embedding_faq",
            ]
            for col in embedding_columns:
                model_dict[col] = model_dict[col].tolist()
        else:
            # Remove embedding columns if not included
            embedding_columns = [
                "embedding_package_name",
                "embedding_package_picture",
                "embedding_url",
                "embedding_installment_month",
                "embedding_installment_limit",
                "embedding_shop_name",
                "embedding_category",
                "embedding_category_tags",
                "embedding_preview_1_10",
                "embedding_selling_point",
                "embedding_meta_keywords",
                "embedding_brand",
                "embedding_min_max_age",
                "embedding_locations",
                "embedding_meta_description",
                "embedding_price_details",
                "embedding_package_details",
                "embedding_important_info",
                "embedding_payment_booking_info",
                "embedding_general_info",
                "embedding_early_signs_for_diagnosis",
                "embedding_how_to_diagnose",
                "embedding_hdcare_summary",
                "embedding_common_question",
                "embedding_know_this_disease",
                "embedding_courses_of_action",
                "embedding_signals_to_proceed_surgery",
                "embedding_get_to_know_this_surgery",
                "embedding_comparisons",
                "embedding_getting_ready",
                "embedding_recovery",
                "embedding_side_effects",
                "embedding_review_4_5_stars",
                "embedding_brand_option_in_thai_name",
                "embedding_faq",
            ]
            for col in embedding_columns:
                del model_dict[col]
        return model_dict

    def to_str_for_broad_rag(self):
        return f"""
    package_name: {self.package_name}
    url: {self.url}?utm_source=ai-chat
    locations: {self.locations}
    price: {self.price}
    brand: {self.brand}
    """

    def to_str_for_narrow_rag(self):
        return f"""
    package_name: {self.package_name}
    package_picture: {self.package_picture}
    url: {self.url}?utm_source=ai-chat
    price: {self.price}
    cash_discount: {self.cash_discount}
    installment_month: {self.installment_month}
    price_after_cash_discount: {self.price_after_cash_discount}
    installment_limit: {self.installment_limit}
    price_to_reserve_for_this_package: {self.price_to_reserve_for_this_package}
    shop_name: {self.shop_name}
    category: {self.category}
    category_tags: {self.category_tags}
    preview_1_10: {self.preview_1_10}
    selling_point: {self.selling_point}
    meta_keywords: {self.meta_keywords}
    brand: {self.brand}
    min_max_age: {self.min_max_age}
    locations: {self.locations}
    meta_description: {self.meta_description}
    price_details: {self.price_details}
    package_details: {self.package_details}
    important_info: {self.important_info}
    payment_booking_info: {self.payment_booking_info}
    general_info: {self.general_info}
    early_signs_for_diagnosis: {self.early_signs_for_diagnosis}
    how_to_diagnose: {self.how_to_diagnose}
    hdcare_summary: {self.hdcare_summary}
    common_question: {self.common_question}
    know_this_disease: {self.know_this_disease}
    courses_of_action: {self.courses_of_action}
    signals_to_proceed_surgery: {self.signals_to_proceed_surgery}
    get_to_know_this_surgery: {self.get_to_know_this_surgery}
    comparisons: {self.comparisons}
    getting_ready: {self.getting_ready}
    recovery: {self.recovery}
    side_effects: {self.side_effects}
    review_4_5_stars: {self.review_4_5_stars}
    brand_option_in_thai_name: {self.brand_option_in_thai_name}
    brand_ranking_position: {self.brand_ranking_position}
    faq: {self.faq}
    """

    def to_str_for_embedding_package_name(self):
        return f"Package Name: {self.package_name}" if self.package_name else ""

    def to_str_for_embedding_package_picture(self):
        return f"Package Picture: {self.package_picture}" if self.package_picture else ""

    def to_str_for_embedding_url(self):
        return f"URL: {self.url}" if self.url else ""

    def to_str_for_embedding_installment_month(self):
        return f"Installment Month: {self.installment_month}" if self.installment_month else ""

    def to_str_for_embedding_installment_limit(self):
        return f"Installment Limit: {self.installment_limit}" if self.installment_limit else ""

    def to_str_for_embedding_shop_name(self):
        return f"Shop Name: {self.shop_name}" if self.shop_name else ""

    def to_str_for_embedding_category(self):
        return f"Category: {self.category}" if self.category else ""

    def to_str_for_embedding_category_tags(self):
        return f"Category Tags: {self.category_tags}" if self.category_tags else ""

    def to_str_for_embedding_preview_1_10(self):
        return f"Preview 1-10: {self.preview_1_10}" if self.preview_1_10 else ""

    def to_str_for_embedding_selling_point(self):
        return f"Selling Point: {self.selling_point}" if self.selling_point else ""

    def to_str_for_embedding_meta_keywords(self):
        return f"Meta Keywords: {self.meta_keywords}" if self.meta_keywords else ""

    def to_str_for_embedding_brand(self):
        return f"Brand: {self.brand}" if self.brand else ""

    def to_str_for_embedding_min_max_age(self):
        return f"Min-Max Age: {self.min_max_age}" if self.min_max_age else ""

    def to_str_for_embedding_locations(self):
        return f"Locations: {self.locations}" if self.locations else ""

    def to_str_for_embedding_meta_description(self):
        return f"Meta Description: {self.meta_description}" if self.meta_description else ""

    def to_str_for_embedding_price_details(self):
        return f"Price Details: {self.price_details}" if self.price_details else ""

    def to_str_for_embedding_package_details(self):
        return f"Package Details: {self.package_details}" if self.package_details else ""

    def to_str_for_embedding_important_info(self):
        return f"Important Info: {self.important_info}" if self.important_info else ""

    def to_str_for_embedding_payment_booking_info(self):
        return f"Payment Booking Info: {self.payment_booking_info}" if self.payment_booking_info else ""

    def to_str_for_embedding_general_info(self):
        return f"General Info: {self.general_info}" if self.general_info else ""

    def to_str_for_embedding_early_signs_for_diagnosis(self):
        return f"Early Signs for Diagnosis: {self.early_signs_for_diagnosis}" if self.early_signs_for_diagnosis else ""

    def to_str_for_embedding_how_to_diagnose(self):
        return f"How to Diagnose: {self.how_to_diagnose}" if self.how_to_diagnose else ""

    def to_str_for_embedding_hdcare_summary(self):
        return f"Hdcare Summary: {self.hdcare_summary}" if self.hdcare_summary else ""

    def to_str_for_embedding_common_question(self):
        return f"Common Question: {self.common_question}" if self.common_question else ""

    def to_str_for_embedding_know_this_disease(self):
        return f"Know This Disease: {self.know_this_disease}" if self.know_this_disease else ""

    def to_str_for_embedding_courses_of_action(self):
        return f"Courses of Action: {self.courses_of_action}" if self.courses_of_action else ""

    def to_str_for_embedding_signals_to_proceed_surgery(self):
        return (
            f"Signals to Proceed Surgery: {self.signals_to_proceed_surgery}" if self.signals_to_proceed_surgery else ""
        )

    def to_str_for_embedding_get_to_know_this_surgery(self):
        return f"Get to Know This Surgery: {self.get_to_know_this_surgery}" if self.get_to_know_this_surgery else ""

    def to_str_for_embedding_comparisons(self):
        return f"Comparisons: {self.comparisons}" if self.comparisons else ""

    def to_str_for_embedding_getting_ready(self):
        return f"Getting Ready: {self.getting_ready}" if self.getting_ready else ""

    def to_str_for_embedding_recovery(self):
        return f"Recovery: {self.recovery}" if self.recovery else ""

    def to_str_for_embedding_side_effects(self):
        return f"Side Effects: {self.side_effects}" if self.side_effects else ""

    def to_str_for_embedding_review_4_5_stars(self):
        return f"Review 4-5 Stars: {self.review_4_5_stars}" if self.review_4_5_stars else ""

    def to_str_for_embedding_brand_option_in_thai_name(self):
        return f"Brand Option in Thai Name: {self.brand_option_in_thai_name}" if self.brand_option_in_thai_name else ""

    def to_str_for_embedding_faq(self):
        return f"FAQ: {self.faq}" if self.faq else ""


# Define HNSW indices to support vector similarity search for each embedding column
indices = [
    Index(
        "hnsw_index_for_embedding_package_name",
        Item.embedding_package_name,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_package_name": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_package_picture",
        Item.embedding_package_picture,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_package_picture": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_url",
        Item.embedding_url,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_url": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_installment_month",
        Item.embedding_installment_month,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_installment_month": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_installment_limit",
        Item.embedding_installment_limit,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_installment_limit": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_shop_name",
        Item.embedding_shop_name,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_shop_name": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_category",
        Item.embedding_category,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_category": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_category_tags",
        Item.embedding_category_tags,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_category_tags": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_preview_1_10",
        Item.embedding_preview_1_10,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_preview_1_10": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_selling_point",
        Item.embedding_selling_point,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_selling_point": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_meta_keywords",
        Item.embedding_meta_keywords,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_meta_keywords": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_brand",
        Item.embedding_brand,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_brand": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_min_max_age",
        Item.embedding_min_max_age,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_min_max_age": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_locations",
        Item.embedding_locations,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_locations": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_meta_description",
        Item.embedding_meta_description,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_meta_description": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_price_details",
        Item.embedding_price_details,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_price_details": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_package_details",
        Item.embedding_package_details,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_package_details": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_important_info",
        Item.embedding_important_info,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_important_info": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_payment_booking_info",
        Item.embedding_payment_booking_info,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_payment_booking_info": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_general_info",
        Item.embedding_general_info,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_general_info": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_early_signs_for_diagnosis",
        Item.embedding_early_signs_for_diagnosis,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_early_signs_for_diagnosis": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_how_to_diagnose",
        Item.embedding_how_to_diagnose,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_how_to_diagnose": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_hdcare_summary",
        Item.embedding_hdcare_summary,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_hdcare_summary": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_common_question",
        Item.embedding_common_question,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_common_question": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_know_this_disease",
        Item.embedding_know_this_disease,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_know_this_disease": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_courses_of_action",
        Item.embedding_courses_of_action,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_courses_of_action": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_signals_to_proceed_surgery",
        Item.embedding_signals_to_proceed_surgery,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_signals_to_proceed_surgery": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_get_to_know_this_surgery",
        Item.embedding_get_to_know_this_surgery,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_get_to_know_this_surgery": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_comparisons",
        Item.embedding_comparisons,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_comparisons": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_getting_ready",
        Item.embedding_getting_ready,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_getting_ready": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_recovery",
        Item.embedding_recovery,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_recovery": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_side_effects",
        Item.embedding_side_effects,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_side_effects": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_review_4_5_stars",
        Item.embedding_review_4_5_stars,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_review_4_5_stars": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_brand_option_in_thai_name",
        Item.embedding_brand_option_in_thai_name,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_brand_option_in_thai_name": "vector_ip_ops"},
    ),
    Index(
        "hnsw_index_for_embedding_faq",
        Item.embedding_faq,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding_faq": "vector_ip_ops"},
    ),
]
