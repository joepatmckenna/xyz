<script>
	export let data;
</script>

<ul>
	{#each data.posts as post}
		<li>
			{post.meta.date}
			<a href={post.path}>
				{post.meta.title}
			</a>
		</li>
	{/each}
</ul>
