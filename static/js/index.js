document.addEventListener("DOMContentLoaded", function() {
  function copyBibtex() {
    const bibtex = `@inproceedings{bai2024sequential,
      title={Sequential modeling enables scalable learning for large vision models},
      author={Bai, Yutong and Geng, Xinyang and Mangalam, Karttikeya and Bar, Amir and Yuille, Alan L and Darrell, Trevor and Malik, Jitendra and Efros, Alexei A},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={22861--22872},
      year={2024}
    }`;
    navigator.clipboard.writeText(bibtex).then(function() {
      alert('BibTeX copied to clipboard!');
    }, function() {
      alert('Failed to copy BibTeX.');
    });
  }

  // Ensure the function is globally accessible
  window.copyBibtex = copyBibtex;
});