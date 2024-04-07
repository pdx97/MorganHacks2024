import React, { useState, useEffect } from 'react';

const JobList = () => {
  const [jobs, setJobs] = useState({});
  const [expandedJob, setExpandedJob] = useState(null);

  // Fetch job data from the Django backend
  useEffect(() => {
    fetch('/api/jobs/')
      .then(response => response.json())
      .then(data => setJobs(data))
      .catch(error => console.error('Error fetching jobs:', error));
  }, []);

  // Toggle job expansion
  const toggleJobExpansion = (jobTitle) => {
    setExpandedJob(expandedJob === jobTitle ? null : jobTitle);
  };

  return (
    <div>
      {Object.entries(jobs).map(([jobTitle, skills]) => (
        <div key={jobTitle}>
          <button onClick={() => toggleJobExpansion(jobTitle)}>
            {jobTitle}
          </button>
          {expandedJob === jobTitle && (
            <ul>
              {skills.map(skill => (
                <li key={skill}>{skill}</li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </div>
  );
};

export default JobList;
