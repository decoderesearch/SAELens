window.SaeTable = {
  showCode: function (release, saeId) {
    const modal = document.getElementById("codeModal");
    const codeContent = document.getElementById("codeContent");
    codeContent.textContent = `from sae_lens import SAE

release = "${release}"
sae_id = "${saeId}"
sae = SAE.from_pretrained(release, sae_id)`;
    modal.style.display = "block";
    return false; // Prevent default link behavior
  },

  closeCode: function () {
    document.getElementById("codeModal").style.display = "none";
  },

  copyCode: function () {
    const codeContent = document.getElementById("codeContent");
    const textArea = document.createElement("textarea");
    textArea.value = codeContent.textContent;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    document.body.removeChild(textArea);

    // Show feedback
    const copyButton = document.querySelector(".saetable-copyButton");
    const originalText = copyButton.textContent;
    copyButton.textContent = "Copied!";
    setTimeout(() => {
      copyButton.textContent = originalText;
    }, 2000);
  },

  selectCode: function (element) {
    const range = document.createRange();
    range.selectNodeContents(element);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  },
};

// Close modal when clicking the X or clicking outside the modal
window.addEventListener("click", function (event) {
  if (event.target.classList.contains("saetable-close")) {
    document.getElementById("codeModal").style.display = "none";
    return;
  }
  const modal = document.getElementById("codeModal");
  if (event.target == modal) {
    modal.style.display = "none";
  }
});
