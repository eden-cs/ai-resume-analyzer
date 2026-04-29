import type { Feedback } from '../types';
import KeywordPill from './KeywordPill';

// Props this component receives from the ResultsPage.
interface Props {
    feedback: Feedback // The feedback object with the high, medium, and low levels.
}

// Priority levels for the feedback.
const priorities = [
    { key: "high", label: "High priority", dotColor: "bg-espresso"},
    { key: "medium", label: "Medium priority", dotColor: "bg-medium-brown"},
    { key: "low", label: "Low priority", dotColor: "bg-tan"},
] as const

export default function FeedbackCard({ feedback }: Props) {
    return (
        // Card container.
        <div className = "bg-taupe border border-espresso/12 rounded-xl p-5 mb-3">

        {/* Card label */}
        <p className = "font-body font-medium text-[10-px] tracking-[0.13em] uppercase text-espresso opacity-60 mb-3">
            Feedback
        </p>

        {/* Map over priorities array to render each section. */}
        {priorities.map((priority, index) => {
            // Get the feedback data for this priority level.
            const level = feedback[priority.key]

            return (
                <div key = {priority.key}>

                    {/* Divider between sections. */}
                    {index > 0 && (
                        <div className = "h-[0.5px] bg-espresso/12 my-3" />
                    )}

                    {/* Priority dot and label row. */}
                    <div className = "flex items-center gap-2 mb-1.5">
                        {/* Colored dot. */}
                        <div className = {`w-[7px] h-[7px] rounded-full flex-shrink-0 ${priority.dotColor}`} />
                        {/* Priority label. */}
                        <p className="font-body font-medium text-[10px] tracking-[0.1em] uppercase text-espresso">
                            {priority.label}
                        </p>
                    </div>

                    {/* Feedback message for this priority level. */}
                    <p className = "font-body font-light text-[12px] text-espresso opacity-70 mb-2">
                        {level.message}
                    </p>

                    {/* Only render keyword pills if there are keywords at this level. */}
                    {level.keywords.length > 0 && (
                        <div className = "flex flex-wrap gap-1.5">
                            {/* Map over keywords and render a pill for each one */}
                            {level.keywords.map((keyword) => (
                                <KeywordPill key = {keyword} keyword = {keyword} />
                            ))}
                        </div>
                    )}
                </div>
            )
        })}
        </div>
    )
}