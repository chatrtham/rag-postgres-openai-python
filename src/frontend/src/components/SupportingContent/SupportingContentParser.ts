import DOMPurify from "dompurify";

type ParsedSupportingContentItem = {
    title: string;
    content: string;
};

export function parseSupportingContentItem(package: string): ParsedSupportingContentItem {
    // Assumes the package starts with the file name followed by : and the content.
    // Example: "sdp_corporate.pdf: this is the content that follows".
    const parts = package.split(": ");
    const title = parts[0];
    const content = DOMPurify.sanitize(parts.slice(1).join(": "));

    return {
        title,
        content
    };
}
