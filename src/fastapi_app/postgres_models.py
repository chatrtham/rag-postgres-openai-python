from __future__ import annotations

from dataclasses import asdict

from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column


# Define the models
class Base(DeclarativeBase, MappedAsDataclass):
    pass


class Package(Base):
    __tablename__ = "packages_all"
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
    faq: Mapped[str] = mapped_column()

    def to_dict(self):
        return asdict(self)

    def to_str_for_broad_rag(self):
        return f"""
    package_name: {self.package_name}
    url: {self.url}
    locations: {self.locations}
    price: {self.price}
    """

    def to_str_for_narrow_rag(self):
        return f"""
    package_name: {self.package_name}
    package_picture: {self.package_picture}
    url: {self.url}
    price: {self.price}
    cash_discount: {self.cash_discount}
    installment_month: {self.installment_month}
    price_after_cash_discount: {self.price_after_cash_discount}
    installment_limit: {self.installment_limit}
    price_to_reserve_for_this_package: {self.price_to_reserve_for_this_package}
    shop_name: {self.shop_name}
    category_tags: {self.category_tags}
    preview_1_10: {self.preview_1_10}
    selling_point: {self.selling_point}
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
    faq: {self.faq}
    """
