// Props this component receives from the ResultsPage.
interface Props {
    suggestions: string // AI generated suggestions string from Gemini API.
}

export default function SuggestionsList({ suggestions }: Props) {
    // Parse the suggestions string into an array of individual suggestion strings.
    const lines = (() => {
        // Handle the case where Gemini returns a JSON array.
        try {
            const parsed = JSON.parse(suggestions)
            // Make sure it's an array of strings.
            if (Array.isArray(parsed)) {
                return parsed as string[]
            }
            // Catch the case where it plain text.
        } catch {
            // Fall through to text splitting.
        }

        // Fall back to splitting by newline and cleaning up numbering or dashes.
        return suggestions
            .split('\n')
            .map((line) =>
                line
                    .replace(/^\d+\.\s*/, "")   // Removes "1. " from the start.
                    .replace(/^-\s*/, "")       // Removes "- " from the start.
                    .trim()                     // Removes extra whitespace.
            )
            .filter((line) => line.length > 0)  // Removes empty lines.
    }) ()

    return (
        <div className = "bg-taupe border border-espresso/12 rounded-xl p-5 mb-3">
           
            {/* Card label */}
            <p className = "font-body font-medium text-[10-px] tracking-[0.13em] uppercase text-espresso opacity-60 mb-3">
                Suggestions
            </p>

            {/* List of suggestion bullet points */}
            <ul className = "flex flex-col gap-2.5">
                {lines.map((suggestion, index) => (
                    <li key = {index} className = "flex items-start gap-2.5">

                        {/* Bullet point */}
                        <div className = "w-1.5 h-1.5 rounded-full bg-medium-brown flex-shrink-0 mt-[7px]" />
                        
                        {/* Suggestion text */}
                        <p className = "font-body text-[13px] text-espresso leading-relaxed">
                            {suggestion}
                        </p>
                    </li>
                ))}
            </ul>
        </div>
    )
}