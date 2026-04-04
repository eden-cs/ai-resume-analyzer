import { useRef } from "react";

// Props this component receives from UploadPage
interface Props {
  file: File | null; // currently selected file
  onFileChange: (file: File | null) => void; // called when file is selected or dropped
  hasError: boolean; // whether to show error state
}

export default function DropZone({ file, onFileChange, hasError }: Props) {
  // Creates a reference to the hidden file input element
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Called when user clicks the drop zone - triggers click on hidden file input
  const handleClick = () => {
    fileInputRef.current?.click(); // only calls .click() if inputRef.current exists
  };

  // Prevent default browser behavior to allow dropping files
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  // Gets the dropped file and passes it up to UploadPAge via onFileChange
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();

    // Get the first file from the dropped files
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      onFileChange(droppedFile);
    }
  };

  return (
    // Drop zone container - styling changes based on hasError prop
    <div
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className={`w-full rounded-[13px] p-8 text-center cursor-pointer transition-colors mb-[10px] ${
        hasError
          ? "bg-error-bg border-[1.5px] border-dashed border-error"
          : "bg-taupe border-[1.5px] border-dashed border-espresso/25 hover:border-espresso/50"
      }`}
    >
      {/* Doc icon */}
      <div className="flex justify-center mb-3">
        <div className="w-7 h-[34px] bg-[#FAF7F2] border border-espresso/12 rounded-[4px] flex flex-col justify-end p-[5px] gap-[3px]">
          {/* Line colors change based on error state */}
          <div
            className={`h-[2px] rounded-sm ${hasError ? "bg-error opacity-40" : "bg-medium-brown"}`}
          />
          <div
            className={`h-[2px] rounded-sm ${hasError ? "bg-error opacity-40" : "bg-medium-brown"}`}
          />
          <div
            className={`h-[2px] w-[60%] rounded-sm ${hasError ? "bg-error opacity-40" : "bg-medium-brown"}`}
          />
        </div>
      </div>

      {/* Main text - shows filw name, error text, or default based on state */}
      <p
        className={`font-body font-medium text-[13px] ${hasError ? "text-error" : "text-espresso"}`}
      >
        {
          file
            ? file.name // show filename when file is selected
            : hasError
              ? "Please upload your resume" // show error message when hasError is true
              : "Drop your resume here" // default text
        }
      </p>

      {/* Subtext */}
      <p className="font-body text-[11px] text-espresso opacity-50 mt-1">
        PDF or DOCX accepted · max 5MB
      </p>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf, .docx"
        className="hidden"
        onChange={(e) => onFileChange(e.target.files?.[0] || null)}
      />
    </div>
  );
}
